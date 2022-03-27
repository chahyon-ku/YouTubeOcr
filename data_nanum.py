
import os
import data_clova


if __name__ == '__main__':
    font_paths = []
    for root, dir, files in os.walk('fonts/nanum'):
        for file in files:
            if file[-4:] == '.ttf' and file[:11] != 'NanumSquare':
                font_paths.append((root, file))
    print(font_paths)
    data_clova.generate_data(font_paths, 64)
