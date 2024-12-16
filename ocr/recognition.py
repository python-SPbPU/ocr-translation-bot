import easyocr
from image_opener import ImageOpener


def is_on_same_line(y1, y2, threshold=10):
    return abs(y1 - y2) < threshold


def recognize_text(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    strings = []
    i = 0
    for bbox, text, prob in results:
        tl, tr, br, bl = bbox
        # print(tl, text)
        if len(strings) == 0:
            strings.append((tl, text))
            i += 1
        else:
            if is_on_same_line(tl[1], strings[i - 1][0][1]):
                strings[i - 1] = (strings[i - 1][0], strings[i - 1][1] + " " + text)
            else:
                strings.append((tl, text))
                i += 1
    return strings


im_o = ImageOpener("ocr/sample.png")
bw = im_o.process_image(low_threshold=150, high_threshold=255, clip_limit=30.0, tile_grid_size=(8, 8))
im_o.write_image(bw, "bw.png")

strings = recognize_text(bw)

for string in strings:
    print(string[1])

