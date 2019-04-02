import csv
import logging
import pandas
from surprise import SVD, Dataset, reader
from surprise.model_selection import cross_validate

logging.basicConfig(format='[%(asctime)-15s][%(levelname)s][%(name)s] : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# File : idPerson, idYoutube
def pre_process(input_file, output_file):
    # Group by user id
    logger.info("Read input file : %s", input_file)
    raw_csv = pandas.read_csv(input_file)
    user_groups = raw_csv.groupby(['user'])
    result_df = pandas.DataFrame(columns=['A'])
    with open(output_file, 'w') as csv_output:
        user_index = 0
        for user_id, user_history in user_groups:
            song_occurences = user_history['song'].apply(lambda i : i.strip()).value_counts()
            song_occurences = song_occurences.to_frame('count').reset_index()
            song_occurences['user_id'] = user_id
            song_occurences.rename(columns={'index': 'song_id'}, inplace=True)
            max_count = song_occurences['count'].max()
            song_occurences['count'] =  song_occurences['count'].apply(lambda i: int((i * 5)/ max_count))
            csv_result = song_occurences.to_csv(index=False, header=(user_index == 0))
            csv_output.write(csv_result)
            logger.info("Writing csv (%d)", user_index)
            user_index = user_index + 1

def predict(csv_file):
    # Use the famous SVD algorithm.
    file_reader = reader.Reader(line_format='item rating user', sep=',', rating_scale=(1, 5), skip_lines=1)
    data = Dataset.load_from_file(file_path=csv_file, reader=file_reader)
    algo = SVD()

    # Run 5-fold cross-validation and print results.
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


def run():
    pre_process(input_file='logs.csv', output_file='users-rating.csv')
    predict('users-rating.csv')

if __name__ == "__main__":
    run()

    