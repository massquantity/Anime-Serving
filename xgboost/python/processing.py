import re
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split


def name2vector(data):
    class NameCorpus(object):
        def __init__(self):
            self.pattern = re.compile(r"\w+")
            self.name= data.name.tolist()

        def __iter__(self):
            for i, line in enumerate(self.name):
                yield TaggedDocument(re.findall(self.pattern, line), [i])

    gensim_model = Doc2Vec(NameCorpus(), vector_size=20, window=5, min_count=1, workers=4)
    name_vectors = pd.DataFrame(gensim_model.docvecs.vectors_docs)
    name_vectors.columns = [f"name_vectors_{i+1}" for i in range(20)]
    return name_vectors


def one_and_multihot(anime):
    anime.genre = anime.genre.str.strip(" ").replace("\s+", "", regex=True).str.lower()
    genre_dummies = anime.genre.str.get_dummies(",")
    type_dummies = pd.get_dummies(anime.type)
    return genre_dummies, type_dummies


def merge_data(rating, anime, type_dummies, genre_dummies, name_vectors):
    merged_anime = pd.concat(
        [anime, type_dummies, genre_dummies, name_vectors], axis=1
    ).drop(["name", "genre", "type"], axis=1)

    data = rating.merge(merged_anime, how="inner", on="anime_id")
    data.drop(["user_id", "anime_id"], axis=1, inplace=True)
    print("data shape: ", data.shape)
    return data


if __name__ == "__main__":
    rating = pd.read_csv("/home/massquantity/Workspace/serving-example/spark/src/main/resources/rating.csv")
    rating = rating[rating.rating != -1].sample(frac=0.1, replace=False)
    anime = pd.read_csv("/home/massquantity/Workspace/serving-example/spark/src/main/resources/anime.csv")
    anime.rename(columns={"rating": "web_rating"}, inplace=True)
    anime.fillna(
        value={
            "genre": "Missing",
            "type": "Missing",
            "web_rating": anime.web_rating.median()
        },
        inplace=True
    )

    print("check missing values: \n")
    for col in anime.columns:
        print(f"{col}: {anime[col].isnull().sum()}")

    name_vectors = name2vector(anime)
    genre_dummies, type_dummies = one_and_multihot(anime)
    data = merge_data(rating, anime, type_dummies, genre_dummies, name_vectors)

    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data.to_csv("../train_data1111.csv", header=None, index=False)
    test_data.to_csv("../test_data1111.csv", header=None, index=False)











