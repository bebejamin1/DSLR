
import fire

from dslr.logistique.regression import AlgoDG, predict
from dslr.parsing.parser import parser
from dslr.visual.describe import describe


class DSLR:
    def __init__(self):
        self.trainpath = "data/datasets/dataset_train.csv"
        self.testpath = "data/datasets/dataset_test.csv"

# ============================= describe ======================================

    def describe(self, filepath) -> None:
        describe(parser(filepath))

# =========================== logreg_train ====================================

    def logreg_train(self) -> None:  # a faire avec le dataset_train # noqa

        data = parser(self.trainpath)

        algo = AlgoDG(data)
        algo.train()

# ========================== logreg_predict ===================================

    def logreg_predict(self) -> None:  # a faire avec le dataset_test

        results = predict(1)

        for algo, prediction in results:
            print(f"Pour l'algorithm {algo} la prediction est : {prediction}"
                  + "\n")


if __name__ == "__main__":
    rs = "\033[0m"
    r = "\033[31m\033[5m\033[1m"
    try:
        fire.Fire(DSLR)
    except Exception as e:
        print(f"{r}[ERROR]:{rs}" + "\n" + f"{e}")
