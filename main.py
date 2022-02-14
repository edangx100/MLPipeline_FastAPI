# Put the code for your API here.
import argparse
import src.data_cleaning
import src.train_test_model
import src.slice_score
import logging


def execute(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "clean_data":
        logging.info("Data cleaning started")
        src.data_cleaning.clean_data()

    if args.action == "all" or args.action == "train_test_model":
        logging.info("Model training/testing started")
        src.train_test_model.train_test_model()

    if args.action == "all" or args.action == "slice_score":
        logging.info("Slice Scoring started")
        src.slice_score.get_scores()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["clean_data",
                 "train_test_model",
                 "slice_score",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    execute(main_args)