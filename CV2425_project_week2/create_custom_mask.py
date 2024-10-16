import cv2
import numpy as np

def create_custom_mask(src, dst):
    # Source img variables
    polygon_points = []
    drawing = False
    polygon_closed = False

    # Dst img variables
    dragging = False
    mask_position = None
    offset_x, offset_y = 0, 0

    def mouse_callback_src(event, x, y, flags, param):
        nonlocal polygon_points, drawing, src_copy, polygon_closed

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to polygon with left click
            polygon_points.append((x, y))
            drawing = True
            
            # Draw lines
            if len(polygon_points) > 1:
                cv2.line(src_copy, polygon_points[-2], polygon_points[-1], (255, 0, 0), 2)
            
            # Draw point
            cv2.circle(src_copy, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Source image", src_copy)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_image = src_copy.copy()
            if len(polygon_points) > 0:
                cv2.line(temp_image, polygon_points[-1], (x, y), (255, 0, 0), 2)
            cv2.imshow("Source image", temp_image)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Close polygon with right click
            drawing = False
            if len(polygon_points) > 2:
                cv2.line(src_copy, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)
                cv2.fillPoly(src_copy, [np.array(polygon_points)], (0, 255, 0))  # Opción de rellenar el polígono
                cv2.imshow("Source image", src_copy)
                polygon_closed = True

    def mouse_callback_dst(event, x, y, flags, param):
        nonlocal dragging, offset_x, offset_y, mask_position, dst_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drag
            dragging = True
            offset_x, offset_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            # Update position and image
            dx, dy = x - offset_x, y - offset_y
            mask_position = (dx, dy)

            temp_image = dst_copy.copy()
            h, w = cropped_mask.shape[:2]
            
            if mask_position:
                tx, ty = mask_position
                tx = max(0, min(tx, dst_copy.shape[1] - w))  
                ty = max(0, min(ty, dst_copy.shape[0] - h))

                # put mask with 3 channels
                mask_3d = cropped_mask[:, :, np.newaxis]
                mask_3d = np.repeat(mask_3d, 3, axis=2)
                
                temp_image[ty:ty+h, tx:tx+w] = np.where(mask_3d == 255, 255, temp_image[ty:ty+h, tx:tx+w])

            cv2.imshow("Destination image", temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            # Drop
            dragging = False
            offset_x, offset_y = x, y

    src_copy = src.copy()
    dst_copy = dst.copy()

    cv2.namedWindow("Source image")
    cv2.setMouseCallback("Source image", mouse_callback_src)

    while True:
        cv2.imshow("Source image", src_copy)
        key = cv2.waitKey(1) & 0xFF
        if polygon_closed or key == 27: #ESC o close polygon
            break

    # Create mask with the polygon
    mask_src = np.zeros(src.shape[:2], dtype=np.uint8)
    if len(polygon_points) > 2:
        cv2.fillPoly(mask_src, [np.array(polygon_points)], 255)

    # cut polygon area
    x_min, y_min = np.min(polygon_points, axis=0)
    x_max, y_max = np.max(polygon_points, axis=0)
    cropped_mask = mask_src[y_min:y_max, x_min:x_max]

    cv2.imshow('Source mask', mask_src); cv2.waitKey(0)

    cv2.namedWindow("Destination image")
    cv2.setMouseCallback("Destination image", mouse_callback_dst)

    while True:
        cv2.imshow("Destination image", dst_copy)
        key = cv2.waitKey(1) & 0xFF
        if not dragging and mask_position: #drag and drop
            break

    mask_dst = np.zeros(dst.shape[:2], dtype=np.uint8)
    if mask_position:
        tx, ty = mask_position

        # polygon points in the new location
        shifted_polygon = np.array(polygon_points) + [tx - x_min, ty - y_min]
        shifted_polygon = shifted_polygon.astype(np.int32)

        #boundaries
        shifted_polygon[:, 0] = np.clip(shifted_polygon[:, 0], 0, dst.shape[1] - 1)
        shifted_polygon[:, 1] = np.clip(shifted_polygon[:, 1], 0, dst.shape[0] - 1)

        #fill polygon
        cv2.fillPoly(mask_dst, [shifted_polygon], 255)

    cv2.imshow('Destination mask', mask_dst); cv2.waitKey(0)

    cv2.destroyAllWindows()

    return mask_src, mask_dst