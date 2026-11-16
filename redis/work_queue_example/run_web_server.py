# import the necessary packages
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import settings
import helpers
import logger
import flask
import redis
import logging
import uuid
import time
import json
import io

logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
	format="%(asctime)s.%(msecs)03d [%(name)s pid=%(process)d] %(message)s")
log = logging.getLogger("web")

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD, db=settings.REDIS_DB)

# seconds to wait for the worker's result notification before giving up
RESULT_TIMEOUT = 30

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/")
def homepage():
	return "Welcome to the PyImageSearch Keras REST API!"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# classification
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image,
				(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image}

			# subscribe to keyspace events for our result key BEFORE
			# pushing the work, otherwise the worker could finish and
			# the notification would fire before we were listening
			p = db.pubsub(ignore_subscribe_messages=True)
			p.subscribe("__keyspace@{}__:{}".format(settings.REDIS_DB, k))
			start = time.time()
			queue_len = db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
			log.info("job %s pushed (queue length now %d)", k[:8], queue_len)
			logger.log("job {} pushed (queue length now {})".format(
				k[:8], queue_len))

			# block until Redis tells us the worker SET our key (or we
			# time out), instead of polling
			notified = False
			try:
				deadline = start + RESULT_TIMEOUT
				while time.time() < deadline:
					message = p.get_message(
						timeout=max(0, deadline - time.time()))
					if message is not None and message["data"] == b"set":
						notified = True
						break
			finally:
				p.close()

			if notified:
				log.info("job %s result notification received after %.2fs",
					k[:8], time.time() - start)
				logger.log("job {} result notification received".format(k[:8]))
			else:
				log.warning("job %s no notification after %.2fs",
					k[:8], time.time() - start)
				logger.log("job {} no notification (timed out)".format(k[:8]))

			# the notification only says the key was set, so fetch the
			# result once (this also covers a missed notification)
			output = db.get(k)
			if output is not None:
				# add the output predictions to our data
				# dictionary so we can return it to the client
				data["predictions"] = json.loads(output.decode("utf-8"))

				# delete the result from the database and indicate that
				# the request was a success
				db.delete(k)
				data["success"] = True
				log.info("job %s result returned to client (%.2fs total)",
					k[:8], time.time() - start)
				logger.log("job {} result returned to client".format(k[:8]))
			else:
				log.error("job %s FAILED: no result in Redis, returning "
					"success=false", k[:8])
				logger.log("job {} FAILED: no result in Redis".format(k[:8]))

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
	log.info("starting web service")
	app.run()
