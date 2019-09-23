import multiprocessing
import os
from io import BytesIO
from urllib import request
import pandas as pd
import re
import tqdm
from PIL import Image


def overwrite_urls(df, args):
    def reso_overwrite(url_tail):
        pattern = 's[0-9]+'
        search_result = re.match(pattern, url_tail)

        if not search_result:
            return url_tail
        else:
            return 's{}'.format(args.target_size)

    def join_url(parsed_url, s_reso):
        parsed_url[-2] = s_reso
        return '/'.join(parsed_url)

    parsed_url = df.url.apply(lambda x: x.split('/'))
    train_url_tail = parsed_url.apply(lambda x: x[-2])
    resos = train_url_tail.apply(lambda x: reso_overwrite(x))

    overwritten_df = pd.concat([parsed_url, resos], axis=1)
    overwritten_df.columns = ['url', 's_reso']
    df['url'] = overwritten_df.apply(lambda x: join_url(x['url'], x['s_reso']), axis=1)
    
    return df


def parse_data(df):
    key_url_list = [line[:2] for line in df.values]

    return key_url_list


def download_image(key_url, args):
    (key, url) = key_url
    filename = os.path.join(os.path.join(args.data_dir, args.data_type), 
                            '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return
    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return
    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return
    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return
    try:
        pil_image_resize = pil_image_rgb.resize((args.target_size, args.target_size))
    except:
        print('Warning: Failed to resize image {}'.format(key))
        return
    try:
        pil_image_resize.save(filename, format='JPEG', quality=args.img_quality)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return
    return


def loader(df, args):
    if not os.path.exists(os.path.join(args.data_dir, args.data_type)):
        os.mkdir(os.path.join(args.data_dir, args.data_type))

    key_url_list = parse_data(df)
    pool = multiprocessing.Pool(processes=args.threads)
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list),
                             total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


def main(args):
    # Set directories and load dataframe
    df = pd.read_csv(os.path.join(args.data_dir, args.data_type+'.csv'))
    # Download images with multiprocess
    loader(overwrite_urls(df), args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='/disk/landmark_rec/',
                        help='The directory of all csv files and to save the result.')
    parser.add_argument('--data_type', type=str, 
                        default='train', choice=['train', 'test'], 
                        help='Download training data or testing data.')
    parser.add_argument('--target_size', type=int, 
                        default=197, help='The resolution of download images.')
    parser.add_argument('--img_quality', type=int, 
                        default=90, help='Quality of JPG file.')
    parser.add_argument('--threads', type=int, 
                        default=10, help='Numbers of CPU to parallel.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
