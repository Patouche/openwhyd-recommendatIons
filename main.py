import csv
import logging
from surprise import SVD, Dataset, reader
from surprise.model_selection import cross_validate

logging.basicConfig(format='[%(asctime)-15s][%(levelname)s][%(name)s] : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# File : idPerson, idYoutube
def pre_process(input_file, output_file):
    # Group by user id
    user_mappings = dict()
    with open(input_file, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            user_id = row[0]
            youtube_id = row[1]
            if user_id not in user_mappings:
                user_mappings[user_id] = list()
            user_mappings[user_id].append(youtube_id)
    # Set frequency to define rating
    user_ratings = dict()
    for user_id, youtube_ids in user_mappings.items():
        user_rating = dict([
            (yid, youtube_ids.count(yid))
            for yid in youtube_ids
        ])
        user_ratings[user_id] = dict([
            (yid, 1 + int((c * 4) / max(user_rating.values())))
            for yid, c in user_rating.items()
        ])
        logger.info("User %s rating : %s", user_id, user_ratings[user_id])

    with open(output_file, 'w') as out_csvfile:
        spamwriter = csv.writer(out_csvfile, delimiter=';', quotechar='"')
        for user_id, ratings in user_ratings.items():
            for youtube_id, rate in ratings.items():
                spamwriter.writerow([
                    user_id,
                    youtube_id,
                    rate
                ])

    
    

def predict():
    # Use the famous SVD algorithm.
    algo = SVD()

    # Run 5-fold cross-validation and print results.
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


def run():
    pre_process(input_file="logs.csv", output_file="users-rating.csv")

if __name__ == "__main__":
    run()

    