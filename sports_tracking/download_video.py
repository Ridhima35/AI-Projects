import urllib.request
url = 'https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4'
print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, 'input.mp4')
print('Downloaded input.mp4 successfully.')
