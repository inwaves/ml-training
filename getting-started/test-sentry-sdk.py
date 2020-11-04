import sentry_sdk

sentry_sdk.init(
        dsn="https://examplePublicKey@o0.ingest.sentry.io/0",
        traces_sample_rate=1.0
)

division_by_zero = 1/0