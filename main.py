# Put the code for your API here.
import argparse
import src.data_cleaning
# import src.train_test_model
# import src.check_score
import logging


def execute(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "clean_data":
        logging.info("Data cleaning started")
        src.data_cleaning.clean_data()

    # if args.action == "all" or args.action == "train_test_model":
    #     logging.info("Train/Test model procedure started")
    #     src.train_test_model.train_test_model()

    # if args.action == "all" or args.action == "check_score":
    #     logging.info("Score check procedure started")
    #     src.check_score.check_score()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["clean_data",
                #  "train_test_model",
                #  "check_score",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    execute(main_args)