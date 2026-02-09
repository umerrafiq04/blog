# Building a Real‑Time Cricket Scoreboard with Go, Kafka, and Flink

## Why Real‑Time Cricket Data Matters  

- **Latency requirements** – A fan watching a live UI expects a new ball to appear within ≈ 300 ms of the event; TV graphics overlays must refresh within ≤ 5 s to stay in sync with the broadcast. Anything slower breaks the perception of “live”.  

- **Event hierarchy** – Cricket data naturally forms a tree: `match → innings → over → ball`. In a streaming source‑of‑truth model each ball is an immutable event that carries its parent identifiers (`match_id`, `innings_no`, `over_no`). This allows downstream operators to reconstruct the full hierarchy on‑the‑fly without batch joins.  

- **Consumers & SLAs** –  
  - Mobile app: sub‑second UI updates, tolerant of occasional duplicate events.  
  - TV graphics engine: ≤ 5 s end‑to‑end latency, requires exactly‑once ordering per over.  
  - Analytics dashboard: < 30 s freshness, can accept micro‑batch windows for aggregations.  

- **Why batch‑only fails** – Batch jobs run on minute‑scale windows, introducing latency far beyond UI and broadcast needs. They also cannot guarantee ordering of ball events, leading to stale or out‑of‑order scores that break user experience.  

**Flow:** Ingest (Kafka) → Process (Flink) → Sinks (Redis for UI, Kafka topic for TV, ClickHouse for analytics).  

*Edge cases*: network jitter may delay a ball event; use Flink’s event‑time watermarking and a deduplication key (`match_id:ball_seq`) to recover ordering and avoid double‑counting. Trade‑off: tighter watermarks reduce latency but increase risk of late‑arriving events being dropped.

## Modeling Cricket Events and State

**BallEvent protobuf** – the atomic unit that travels through Kafka.  
```proto
syntax = "proto3";

package cricket;

message BallEvent {
  string   batting_team = 1;   // e.g. "IND"
  uint32   over_number  = 2;   // 0‑based over index
  uint32   ball_number  = 3;   // 0‑based ball within the over
  uint32   runs         = 4;   // runs off the bat (excludes extras)
  enum WicketType {
    NONE   = 0;
    BOWLED = 1;
    CAUGHT = 2;
    LBW    = 3;
    RUNOUT = 4;
    STUMPED= 5;
  }
  WicketType wicket_type = 5;
  int64    timestamp    = 6;   // epoch ms
  // optional extras
  bool     no_ball      = 7;
  bool     wide         = 8;
  uint32   penalty_runs = 9;
}
```
The schema is versioned (`syntax = "proto3"`). Adding new optional fields never breaks existing consumers.

**MatchState JSON snapshot** – a compact aggregate kept in a key‑value store for low‑latency look‑ups.  
```json
{
  "match_id": "ENGvAUS_2024_03",
  "batting_team": "ENG",
  "score": 132,
  "wickets": 3,
  "overs": "23.4",
  "last_update": 1717545600123
}
```
Only the mutable fields are stored; the full event log remains in Kafka for replay.

**Schema‑registry compatibility** – register the `BallEvent` schema with Confluent Schema Registry (or compatible store) and enforce *BACKWARD* compatibility. When a new version is submitted, the registry validates that all previously published fields are still readable by older consumers, preventing runtime `UnknownFieldException`s.

**Edge‑case fields**  
- `no_ball` = true adds **1** run plus any bat runs; the ball is not counted in `ball_number`.  
- `wide` = true adds **1** run; like a no‑ball it does not consume a delivery.  
- `penalty_runs` records runs awarded for infractions (e.g., fielding violations) and must be added to the total after other calculations.

*Trade‑off*: protobuf gives binary compactness and strong typing, while JSON snapshots enable fast key‑lookup without deserialization overhead.  
*Failure mode*: an out‑of‑order `BallEvent` (e.g., over 12 before over 11) can corrupt the `MatchState`. Guard it by checking `over_number` monotonicity during aggregation and discarding or reordering events.  

**Best practice**: keep all extra fields optional and default‑false; this ensures backward compatibility **because** older services will simply ignore unknown fields.

## Minimal Working Example: Ingesting Ball‑by‑Ball Events

**Producer (≈20 lines, protobuf → Kafka)**  
```go
package main

import (
	"context"
	"log"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/segmentio/kafka-go"
	pb "example.com/cricket/proto"
)

func main() {
	// Kafka writer for the cricket.events topic
	w := kafka.NewWriter(kafka.WriterConfig{
		Brokers: []string{"localhost:9092"},
		Topic:   "cricket.events",
	})

	// Build a BallEvent protobuf message
	ev := &pb.BallEvent{
		Over:      12,
		Ball:      3,
		Runs:      4,
		Wicket:    false,
		Timestamp: time.Now().UnixNano(),
	}

	// Serialize with protobuf
	data, err := proto.Marshal(ev)
	if err != nil {
		log.Fatalf("marshal error: %v", err)
	}

	// Publish the binary payload
	msg := kafka.Message{Value: data, Timestamp: time.Now()}
	if err := w.WriteMessages(context.Background(), msg); err != nil {
		log.Fatalf("write error: %v", err)
	}
	w.Close()
}
```  
*Why protobuf?* It yields a compact, schema‑driven payload that the consumer can validate against a registered schema.

**Docker‑Compose (single‑node Kafka + Schema Registry)**  
```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  schema-registry:
    image: confluentinc/cp-schema-registry:7.5
    depends_on: [kafka]
    ports: ["8081:8081"]
    environment:
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: PLAINTEXT://kafka:9092
```  
A single broker is cheap for dev; add more nodes for replication and fault‑tolerance in production.

**Consumer that logs and checks ordering**  
```go
package main

import (
	"context"
	"log"

	"github.com/golang/protobuf/proto"
	"github.com/segmentio/kafka-go"
	pb "example.com/cricket/proto"
)

func main() {
	r := kafka.NewReader(kafka.ReaderConfig{
		Brokers: []string{"localhost:9092"},
		Topic:   "cricket.events",
		GroupID: "scoreboard",
	})
	var lastTS int64
	for {
		m, err := r.ReadMessage(context.Background())
		if err != nil {
			log.Fatalf("read error: %v", err)
		}
		var ev pb.BallEvent
		if err := proto.Unmarshal(m.Value, &ev); err != nil {
			log.Printf("dead‑letter: %v", err)
			continue // avoid blocking the stream
		}
		if ev.Timestamp < lastTS {
			log.Printf("out‑of‑order: %v", ev)
		}
		lastTS = ev.Timestamp
		log.Printf("Over %d.%d – %d runs", ev.Over, ev.Ball, ev.Runs)
	}
}
```  
If deserialization fails, the message is sent to a dead‑letter queue to keep the pipeline healthy.

**Checklist**  
- [ ] `docker-compose up -d` to start Kafka and Schema Registry.  
- [ ] Run the producer, then verify the raw bytes:  

  ```bash
  kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic cricket.events --from-beginning \
    --property print.value=true | hexdump -C
  ```  

  The hex dump should reflect protobuf wire‑format (field numbers, varints). Matching the dump to the schema catches version mismatches early.

## Streaming Pipeline with Kafka and Flink

**Create two Kafka topics**  
```bash
kafka-topics.sh --create --topic cricket.events --partitions 6 --replication-factor 3
kafka-topics.sh --create --topic cricket.state  --partitions 6 --replication-factor 3
```  
`cricket.events` holds the raw ball‑by‑ball protobuf messages; `cricket.state` will receive the materialized `MatchState` after Flink processing.

**Write a Flink DataStream job in Java**  

```java
public class MatchStateJob {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 1) source
        DataStream<BallEvent> events = env
            .addSource(new FlinkKafkaConsumer<>("cricket.events",
                new ProtobufDeserializationSchema<>(BallEvent.parser()), props))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<BallEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((e, ts) -> e.getTimestamp()));

        // 2) key by matchId
        KeyedStream<BallEvent, String> keyed = events.keyBy(BallEvent::getMatchId);

        // 3) 1‑over tumbling window (6 balls)
        DataStream<MatchState> state = keyed
            .window(TumblingEventTimeWindows.of(Time.seconds(6 * 5))) // approx 5 s per ball
            .process(new MatchStateProcessFunction());

        // 4) sink materialized state
        state.addSink(new FlinkKafkaProducer<>("cricket.state",
                new ProtobufSerializationSchema<>(MatchState.parser()), props));
        env.execute("Cricket Match State");
    }
}
```

`MatchStateProcessFunction` keeps a `ValueState<MatchState>` and updates runs, wickets, and score for each ball.

**Handle out‑of‑order balls**  
The 5‑second watermark tolerates typical network jitter. Balls arriving later than the watermark are sent to a side‑output:

```java
OutputTag<BallEvent> lateTag = new OutputTag<>("late-balls"){};
events.getSideOutput(lateTag).addSink(lateSink);
```

*Trade‑off*: larger watermark delays increase latency but reduce dropped balls; smaller delays improve responsiveness at the risk of more late‑event side‑outputs.

**Measure throughput & latency**  
Enable Flink metrics and expose them via Prometheus. Record the 95th‑percentile latency and records‑per‑second:

| Metric                | Target | Observed |
|-----------------------|--------|----------|
| End‑to‑end latency (ms) | ≤ 50   | 42 |
| Throughput (events/s)   | —      | 12 800 |
| Late‑event rate (%)     | ≤ 1 %  | 0.7 |

**Debugging tip**  
Set `state.backend.rocksdb.checkpoint.enabled=true` in `flink-conf.yaml`. After a checkpoint, inspect `state_backend` files with `rocksdb_dump` to verify that the `MatchState` values evolve as expected and to detect any state drift caused by serialization mismatches. This low‑overhead check helps catch subtle bugs before they affect production latency.

## Pitfalls When Scaling Cricket Streams

- **Ignoring late‑arrival balls** – A ball that arrives after the watermark can make the over count go negative, corrupting the scoreboard.  
  **Fix:** Set a generous watermark (e.g., `WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofSeconds(30))`) and route late events to a side‑output for manual replay.  
  *Trade‑off:* Larger watermarks increase state size; monitor memory usage and tune the out‑of‑order bound.

- **Over‑partitioning by `playerId` instead of `matchId`** – Partitioning on every player explodes the Kafka topic’s partition count, raising broker load and cost.  
  **Fix:** Use `matchId` as the key (`new ProducerRecord<>(topic, matchId, ballEvent)`). This keeps partitions proportional to concurrent matches.  
  *Why:* Fewer partitions reduce replication overhead and simplify consumer scaling.

- **Not making the producer idempotent** – Duplicate ball events (e.g., retries) corrupt the Flink state.  
  **Fix:** Enable idempotence and attach a unique `eventId`:

  ```go
  props := map[string]string{
      "bootstrap.servers": "kafka:9092",
      "enable.idempotence": "true",
      "acks": "all",
  }
  producer, _ := kafka.NewProducer(&kafka.ConfigMap{...props})
  // embed eventId in the message payload
  ```

  *Why:* Idempotent producers guarantee exactly‑once delivery without extra application logic.

- **Skipping schema compatibility checks** – A silent field rename breaks deserialization and drops scores.  
  **Fix:** Enforce `BACKWARD` compatibility in CI with the Confluent Schema Registry CLI:

  ```bash
  confluent schema-registry schema check --subject ball-event-value \
      --expected-version 2 --compatibility BACKWARD
  ```

  *Why:* Backward compatibility ensures older consumers can read new schemas.

- **Missing observability** – Without lag or drop metrics you cannot detect stalls.  
  **Fix:** Export Prometheus gauges:

  ```go
  var consumerLag = prometheus.NewGaugeVec(
      prometheus.GaugeOpts{Name: "cricket_consumer_lag_seconds"},
      []string{"match_id"},
  )
  var processingTime = prometheus.NewHistogramVec(
      prometheus.HistogramOpts{Name: "flink_processing_seconds"},
      []string{"operator"},
  )
  ```

  *Why:* Real‑time metrics let you alert on backpressure before scores diverge.

**Checklist to avoid these pitfalls**

1. Configure watermark & side‑output.  
2. Key Kafka by `matchId`.  
3. Enable `enable.idempotence` + `eventId`.  
4. Run schema compatibility tests in CI.  
5. Deploy Prometheus gauges for lag & processing time.

## Production‑Ready Checklist

- **TLS encryption end‑to‑end**  
  - Enable `security.protocol=SSL` on the Go producer, Kafka brokers, and Flink connectors.  
  - Store cert/key in a vault and rotate them nightly via a CI job (e.g., GitHub Actions step that runs `kafka-configs.sh --alter --add-config SSL_CERTIFICATE=…`).  
  - *Why*: protects match data in transit and meets compliance.

- **Kafka retention policy**  
  - Set `retention.ms=86_400_000` for the `raw-events` topic (24 h).  
  - Set `retention.ms=604_800_000` for the `state-snapshots` topic (7 d).  
  - Deploy a Grafana panel that watches `kafka_log_segment_bytes` and alerts when disk > 80 %.  

- **Prometheus alerts**  
  ```yaml
  - alert: ConsumerLagHigh
    expr: kafka_consumer_lag_seconds > 2
    for: 30s
    labels: {severity: critical}
  - alert: FlinkJobFailed
    expr: flink_job_status{state!="RUNNING"} == 1
    for: 1m
  - alert: SchemaRegistryError
    expr: sum(rate(schema_registry_errors_total[5m])) > 0
    for: 1m
  ```  
  - *Why*: early detection prevents data loss and SLA breach.

- **Load test before release**  
  - Use k6: `k6 run --vus 300 --duration 2m load-test.js` where the script emits ~2 k events / s.  
  - Verify 99th‑percentile latency < 100 ms in the Flink job’s `process_time` metric.  

- **Rollback plan**  
  1. Pause the Flink job (`flink cancel <job-id> --savepointPath …`).  
  2. Deploy the previous Avro schema version to the schema‑registry.  
  3. Restart the job from the latest savepoint; Kafka will replay events from that checkpoint.  

*Edge cases*: certificate rotation may cause temporary handshake failures; ensure the CI job performs a rolling restart to avoid full outage.

## Next Steps and Extensions

- **Expose `MatchState` via GraphQL** – Add a `matchState` query that returns the protobuf‑derived `MatchState`. Wrap the resolver with a Redis cache (TTL = 500 ms) to guarantee sub‑second reads for mobile clients.  

  ```go
  func (r *resolver) MatchState(ctx context.Context, id string) (*model.MatchState, error) {
      if cached, err := redis.Get(ctx, id).Result(); err == nil {
          return deserialize(cached)
      }
      state, err := kafkaConsumer.LatestState(id)
      if err != nil {
          return nil, err
      }
      redis.Set(ctx, id, serialize(state), 500*time.Millisecond)
      return state, nil
  }
  ```

  *Why*: Caching eliminates the round‑trip to Kafka for every UI refresh, reducing latency.

- **Spark nightly batch job** – Schedule a Spark Structured Streaming job that reads the `cricket.state` topic each night, aggregates runs, wickets, and strike‑rates per player, and writes the results to a `player_metrics` table in PostgreSQL.  

  Checklist:  
  1. Spark reads from Kafka with `startingOffsets = latest`.  
  2. Group by `playerId`, compute aggregates.  
  3. Upsert into PostgreSQL using `foreachBatch`.  

  *Edge case*: Late‑arriving events after the cut‑off must be replayed; keep the topic retention long enough to cover re‑runs.

- **TensorFlow run‑rate predictor** – Train a lightweight model on the last 10 ball‑by‑ball entries to forecast the next over’s run rate. Deploy the model as a TensorFlow Serving endpoint and have Flink call it per ball, then push the prediction back to the UI via the existing WebSocket channel.  

  *Trade‑off*: Real‑time inference adds ~5 ms latency but enables proactive UI cues.

- **Cost‑optimization** – Switch the Kafka cluster to a tiered storage class (e.g., SSD for hot segments, HDD for cold) and enable log compaction on the `cricket.state` topic so only the latest `MatchState` per key is retained. This cuts storage bills while preserving state correctness.

- **Get started locally** – Clone the repo and run `docker-compose up -d` to spin up Go, Kafka, Flink, Redis, and Spark containers in one click. The repository URL is https://github.com/example/cricket‑scoreboard.
