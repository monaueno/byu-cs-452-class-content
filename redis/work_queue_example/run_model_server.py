# import the necessary packages
from tensorflow.keras.applications import ResNet50
from keras.applications import imagenet_utils
import numpy as np
import settings
import helpers
import logger
import logging
import redis
import time
import json

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
	format="%(asctime)s.%(msecs)03d [%(name)s pid=%(process)d] %(message)s")
log = logging.getLogger("worker")

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD, db=settings.REDIS_DB)

def classify_process():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	log.info("loading model")
	model = ResNet50(weights="imagenet")
	log.info("model loaded, waiting for jobs on '%s'", settings.IMAGE_QUEUE)

	# continually pool for new images to classify
	while True:
		# atomically claim a batch of images from the database (LPOP removes
		# them in the same step, so no other worker can get the same ones),
		# then initialize the image IDs and batch of images themselves
		queue = db.lpop(settings.IMAGE_QUEUE, settings.BATCH_SIZE) or []
		imageIDs = []
		batch = None

		# loop over the queue
		for q in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
			image = helpers.base64_decode_image(q["image"],
				settings.IMAGE_DTYPE,
				(1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
					settings.IMAGE_CHANS))

			# check to see if the batch list is None
			if batch is None:
				batch = image

			# otherwise, stack the data
			else:
				batch = np.vstack([batch, image])

			# update the list of image IDs
			imageIDs.append(q["id"])

		# check to see if we need to process the batch
		if len(imageIDs) > 0:
			# classify the batch
			log.info("popped %d job(s): %s", len(imageIDs),
				", ".join(i[:8] for i in imageIDs))
			logger.log("popped {} job(s): {}".format(len(imageIDs),
				", ".join(i[:8] for i in imageIDs)))
			start = time.time()
			preds = model.predict(batch)
			results = imagenet_utils.decode_predictions(preds)
			log.info("classified batch %s in %.2fs", batch.shape,
				time.time() - start)
			logger.log("classified batch of {}".format(len(imageIDs)))

			# loop over the image IDs and their corresponding set of
			# results from our model
			for (imageID, resultSet) in zip(imageIDs, results):
				# initialize the list of output predictions
				output = []

				# loop over the results and add them to the list of
				# output predictions
				for (imagenetID, label, prob) in resultSet:
					r = {"label": label, "probability": float(prob)}
					output.append(r)

				# store the output predictions in the database, using
				# the image ID as the key so we can fetch the results
				db.set(imageID, json.dumps(output))
				log.info("job %s result saved (top: %s %.4f)", imageID[:8],
					output[0]["label"], output[0]["probability"])
				logger.log("job {} result saved (top: {})".format(
					imageID[:8], output[0]["label"]))

		# sleep for a small amount
		time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
	classify_process()
