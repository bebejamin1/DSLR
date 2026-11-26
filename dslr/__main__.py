
import fire

from dslr.logistique.regression import AlgoDG, predict
from dslr.parsing.parser import parser
from dslr.visual.describe import describe
from dslr.logistique.normaliser import normalise
from dslr.visual.histogram import histogram
from dslr.visual.scatter_plot import scatter_plot
from dslr.visual.pair_plot import pair_plot



class DSLR:
    def __init__(self, trainpath: str = "data/datasets/dataset_train.csv", testpath: str = "data/datasets/dataset_test.csv"):
        self.trainpath = trainpath
        self.testpath = testpath

    def normalise(self, filepath: str = None):
        path = filepath if filepath is not None else self.trainpath
        normalise(parser(path))

# ============================== visual =======================================

    def describe(self) -> None:
        describe(parser(self.trainpath))


    def histogram(self) -> None:
        histogram(parser(self.trainpath))


    def scatter_plot(self) -> None:
        scatter_plot(parser(self.trainpath))


    def pair_plot(self) -> None:
        pair_plot(parser(self.trainpath))


# =========================== logreg_train ====================================

    def logreg_train(self) -> None:  # a faire avec le dataset_train # noqa

        data = parser(self.trainpath)

        algo = AlgoDG(data)
        algo.train()

# ========================== logreg_predict ===================================

    def logreg_predict(self) -> None:  # a faire avec le dataset_test
        from dslr.logistique.regression import predict_data
        from dslr.parsing.parser import parser
        import csv
        
        test_data = parser(self.testpath)
        predictions = predict_data(test_data, "batch_gd")
        
        with open("houses.csv", "w", newline="\n") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Index", "Hogwarts House"])
            for idx, prediction in enumerate(predictions):
                writer.writerow([idx, prediction])
        
        print("Fichier houses.csv généré avec succès!")


if __name__ == "__main__":
    rs = "\033[0m"
    r = "\033[31m\033[5m\033[1m"
    try:
        fire.Fire(DSLR)
    except Exception as e:
        print(f"{r}[ERROR]:{rs}" + "\n" + f"{e}")
