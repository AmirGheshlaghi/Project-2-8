import cv2
import os
import mediapipe as mp

images = []
eyes_locs = []
mean_dist_within = 135
widths = []
heights_up = []
heights_down = []
teta_s = []
rotated_s = []

model = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Calling the names of the photos
KNOW_DIR = 'images'
images_name = os.listdir(KNOW_DIR)

for img_name in images_name:

	# Calling the photos
	img = cv2.imread(KNOW_DIR+'/'+img_name)

	# Find eye landmarks
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	lm = model.process(img_rgb).multi_face_landmarks

	if lm:

		# Determining the coordinates of the eyes
		point0 = lm[0].landmark[130]
		point1 = lm[0].landmark[359]
		lm0 = int(point0.x * img.shape[1]), int(point0.y * img.shape[0])
		lm1 = int(point1.x * img.shape[1]), int(point1.y * img.shape[0])

		# Equalizing the distance between the eyes
		scale = mean_dist_within/abs(lm0[0]-lm1[0])
		img = cv2.resize(img, None, fx=scale, fy=scale)
		lm0 = int(lm0[0]*scale), int(lm0[1]*scale)
		lm1 = int(lm1[0]*scale), int(lm1[1]*scale)
		lm0 = list(lm0)
		lm1 = list(lm1)

		# Cropping the image (horizontally) in order to 
		# place the eyes in the center of the image
		if lm0[0] < (img.shape[1]- lm1[0]):
			img = img[:, :lm1[0]+lm0[0], :]

		else:
			img = img[:, lm0[0]-(img.shape[1]-lm1[0]):, :]
			A = lm0[0]-(img.shape[1]-lm1[0])
			lm0[0] = lm0[0] - A/2
			lm1[0] = lm1[0] - A/2

		# Determining the angle of rotation of the eyes
		cx = (lm1[0]+lm0[0])/2
		cy = (lm1[1]+lm0[1])/2
		r = ((lm0[0]-lm1[0])**2+(lm0[1]-lm1[1])**2)**0.5/2
		r_teta = ((cy-lm1[1])**2+(cx+r-lm1[0])**2)**0.5
		teta = r_teta/r*180/3.14
		if lm0[1]<lm1[1]:
			teta_s.append(teta)

		else:
			teta_s.append(-teta)

		eyes_locs.append([lm0, lm1])
		widths.append(img.shape[1])
		heights_up.append(lm0[1])
		heights_down.append(img.shape[0]-lm0[1])
		images.append(img)

	else:
		print("The location of the eyes was not found!")

# Finding the shortest distance between the 
# left eye and the vertical axis of the photo
min_width = min(widths)

# Finding the shortest distance between the 
# left eye and the upper horizontal axis of the photo
min_height_up = min(heights_up)

# Finding the shortest distance between the 
# left eye and the bottom horizontal axis of the photo
min_height_down = min(heights_down)

for (img, eyes_loc, teta) in zip(images, eyes_locs, teta_s):

	# Equalize the dimensions of the images
	img = img[:, int((img.shape[1]-min_width)/2):int(img.shape[1]-(img.shape[1]-min_width)/2), :]
	img = img[int(eyes_loc[0][1]-min_height_up):eyes_loc[0][1]+min_height_down, :, :]

	# Halve the dimensions of the images
	img = cv2.resize(img, None, fx=0.5, fy=0.5)

	# Rotate the photo so that the eyes are horizontal
	(h, w) = img.shape[:2]
	cx = int(img.shape[1]/2)
	cy = int(min_height_up/2+abs((eyes_loc[0][1]-eyes_loc[1][1])/2))
	M = cv2.getRotationMatrix2D((cx, cy), teta, 1.0)
	rotated = cv2.warpAffine(img, M, (w, h))
	rotated_s.append(rotated)

	# Show pictures
	cv2.imshow("Image", rotated)
	q = cv2.waitKey(0)
	if q == ord('q'):
		break

# Determine the name of the video and the size of the video
height, width, layers = rotated_s[0].shape
video_name = 'Result.avi'
video = cv2.VideoWriter(video_name, 0, 1, (width,height))

# Save the results in video format
for rotated in rotated_s:
	video.write(rotated)

cv2.destroyAllWindows()
video.release()
