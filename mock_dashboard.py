from Adafruit_IO import Client

# For security reasons i did not add my credentials, if you want to test it dm me hehehehhe
AIO_USERNAME = "123USERNAME"
AIO_KEY = "123KEY"

aio = Client(AIO_USERNAME, AIO_KEY)

deer_count = 15
deer_threshold = 20
avg_confidence = 0.85

threshold_data = aio.receive("deer-threshold")
deer_threshold = int(float(threshold_data.value))

# Confirm the names and keys of the available feeds
for feed in aio.feeds():
    print(feed.name, "→", feed.key)

if deer_count >= deer_threshold:
    alert_status = "Warning: Excessive deer count"
else:
    alert_status = "Normal"

print("Sending deer count:", deer_count)

aio.send_data("deer-count", deer_count)
aio.send_data("deer-threshold", deer_threshold)
aio.send_data("alert-status", alert_status)
