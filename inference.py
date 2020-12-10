# using trained model to infer in tests
import sys


def test():
	'''Do prediciton test in given image
	'''
	pass

def draw_result(image_path, eye, heatmap, gaze_point):
	'''Visualize result
	'''
	x1, y1 = eye
	x2, y2 = gaze_point
	im = cv2.imread(image_path)
	image_height, image_width = im.shape[:2]
	x1, y1 = image_width * x1, y1 * image_height
	x2, y2 = image_width * x2, y2 * image_height
	x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
	cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
	cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
	cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 3)

	# heatmap visualization
	heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
	heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
	heatmap = cv2.resize(heatmap, (image_width, image_height))

	heatmap = (0.8 * heatmap.astype(np.float32) + 0.2 * im.astype(np.float32)).astype(np.uint8)
	img = np.concatenate((im, heatmap), axis=1)
	cv2.imwrite('tmp.png', img)
	return img

def main(test_image_path,x,y):
	'''
	load the net first!!!!!
	'''
    heatmap, gaze_x, gaze_y = test(net, test_image_path, (x, y))
	draw_result(test_image_path, (x, y), heatmap, (gaze_x, gaze_y))


if __name__ == '__main__':
	main(sys.argv[1],float(sys.argv[2]),float(sys.argv[3]))
	
	