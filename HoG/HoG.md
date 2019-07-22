# `HoG`

## Feature descriptor

1. common features
   - edge, corners
   - color
2. HoG
   - the **distribution** ( histograms ) of directions of gradients ( oriented gradients ) 

## Algorithms

### Step1 Preprocessing

- a fixed aspect ratio
- gamma correction

### Step2 Calculate gradient

- calculate the horizontal and vertical gradients, then calculate the histogram of gradients 

  ```python
  #Sobel operator
  # Read image
  im = cv2.imread('ball.png')
  im = np.float32(im) / 255.0
   
  # Calculate gradient 
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
  ```

- find the magnitude and direction of gradient 

  ```python
  mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
  ```

### Step3 Calculate Histogram of Gradients in cells

- reason: more robust to noise 
- add the contributions of all the pixels in the 8×8 cells proportionally  to create the 9-bin histogram

### Step4 16×16 Block Normalization

- insensitive to lighting variations

### Step5 Visualize

