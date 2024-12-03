FROM rust:1.80.0 as builder

COPY . .

RUN cargo build --release

FROM ubuntu:22.04

COPY --from=builder /target/release/candle-demo /usr/local/bin/candle-demo

CMD ["candle-demo"]
