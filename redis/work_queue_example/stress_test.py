# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import logging
import requests
import logger
import time

logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
    format="%(asctime)s.%(msecs)03d [%(name)s] %(message)s")
log = logging.getLogger("stress")

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "jemma.png"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
# (each queued image takes ~0.9 MB in Redis, so a free 30 MB database
# only holds about 30 at once; raise this if you run Redis locally)
NUM_REQUESTS = 20
SLEEP_COUNT = 0.05

def call_predict_endpoint(n):
    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    # submit the request
    log.info("request %d sent", n)
    logger.log("request {} sent".format(n))
    start = time.time()
    r = requests.post(KERAS_REST_API_URL, files=payload).json()

    # ensure the request was sucessful
    if r["success"]:
        log.info("request %d OK after %.2fs", n, time.time() - start)
        logger.log("request {} OK".format(n))

        # loop over the predictions and display them
        for (i, result) in enumerate(r["predictions"]):
            print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))


    # otherwise, the request failed
    else:
        log.error("request %d FAILED after %.2fs", n, time.time() - start)
        logger.log("request {} FAILED".format(n))

# loop over the number of threads
threads = []
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.start()
    threads.append(t)
    time.sleep(SLEEP_COUNT)

# wait for every request to finish before exiting
for t in threads:
    t.join()
log.info("all %d requests finished", NUM_REQUESTS)
logger.log("all {} requests finished".format(NUM_REQUESTS))
