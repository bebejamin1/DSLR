
import fire

from dslr.logistique.regression import AlgoDG
from dslr.parsing.parser import parser

class DSLR:

    def describe(self) -> None:
        pass

    def logreg_train(self, filepath = 'data/datasets/dataset_train.csv') -> None:  # a faire avec le dataset_train
        print("cc ma biche lucas")
        parser(filepath)

        # ajouter tri et renvoyer list
        data = []

        AlgoDG.train(data)

    def logreg_predict(self) -> None:  # a faire avec le dataset_test
        print("plus tard a faire apres le train")


if __name__ == "__main__":
    rs = "\033[0m"
    r = "\033[31m\033[5m\033[1m"
    try:
        fire.Fire(DSLR)
    except Exception as e:
        print(f"{r}[ERROR]:{rs}" + "\n" + f"{e}")
