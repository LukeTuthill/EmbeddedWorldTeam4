"""Fit the entire webcam view into the model input without cutting off edges."""


def letterbox(frame, width, height, cv2):
    if width <= 0 or height <= 0:
        raise ValueError("The model must report a positive image input size")
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, round(source_width * scale)))
    resized_height = max(1, min(height, round(source_height * scale)))
    if (source_width, source_height) == (resized_width, resized_height):
        resized = frame
    else:
        resized = cv2.resize(frame, (resized_width, resized_height),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    return cv2.copyMakeBorder(resized, top, height - resized_height - top,
                              left, width - resized_width - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
