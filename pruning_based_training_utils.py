import torch.nn as nn
import sys
from copy import deepcopy
from torch import cuda, max, mean, FloatTensor, no_grad, save, load, device
from torchvision import models
from pandas import DataFrame
from os import listdir, remove
from torch.optim import SGD
from generic_utils import log_progress, send_telegram_message
from torch import optim
import traceback


def load_model_and_optim_state(path):
    """This function loads and returns the model stored
    in the given path.

    Args:
        path (string): path where the checkpoint is stored.

    Returns:
        tuple: pytorch model and optimizer if found, otherwise None, None.
    """
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    hw_device = device("cuda:0" if cuda.is_available() else "cpu")
    n_classes = 120
    n_inputs = model.fc.in_features
    model.aux_logits = False
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, n_classes),
        nn.LogSoftmax(dim=1),
    ).to(hw_device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    try:
        torch_checkpoint = load(path)
        model.load_state_dict(torch_checkpoint["model_state_dict"])
        model.idx_to_class = torch_checkpoint["idx_to_class"]
        model.cuda()
        optimizer.load_state_dict(torch_checkpoint["optimizer_state_dict"])
    except Exception as ex:
        print("CHECKPOINT LOADING EXCEPTION: " + traceback.format_exc())
        return None, None
    return model, optimizer


class training_info_container:
    """Class defined to contain every variable needed by the training
    function, allowing an easy modification of them accross functions.
    """

    pass


def store_model_and_optim_state(tc, path):
    """This function allows to store models' and optimizers' state
    along with the related autocast scaler state as checkpoint.

    Args:
        tc (training_info_container): container of every info related to
        the current training.
        path (string): save path.
    """
    save(
        {
            "model_state_dict": tc.current_model.state_dict(),
            "idx_to_class": tc.current_model.idx_to_class,
            "class_to_idx": tc.current_model.class_to_idx,
            "optimizer_state_dict": tc.current_optimizer.state_dict(),
            "scaler_state_dict": tc.scaler.state_dict(),
        },
        path,
    )


def load_model_and_optim_state_with_scaler(tc, path):
    """Similar to load_model_and_optim_state, this function
    loads also the related scaler from the checkpoint.

    Args:
        tc (training_info_container): container of every info related to
        the current training.
        path (string): load path.
    """
    tc.current_model = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1
    )
    tc.current_model.aux_logits = False
    hw_device = device("cuda:0" if cuda.is_available() else "cpu")
    n_classes = 120
    n_inputs = tc.current_model.fc.in_features
    tc.current_model.fc = nn.Sequential(
        nn.Linear(n_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, n_classes),
        nn.LogSoftmax(dim=1),
    ).to(hw_device)
    tc.current_optimizer = SGD(tc.current_model.parameters(), lr=0.01, momentum=0.9)
    tc.scaler = cuda.amp.GradScaler()
    try:
        torch_checkpoint = load(path)
        tc.current_model.load_state_dict(torch_checkpoint["model_state_dict"])
        tc.current_model.idx_to_class = torch_checkpoint["idx_to_class"]
        tc.current_model.class_to_idx = torch_checkpoint["class_to_idx"]
        tc.current_model.cuda()
        tc.current_optimizer.load_state_dict(torch_checkpoint["optimizer_state_dict"])
        tc.scaler.load_state_dict(torch_checkpoint["scaler_state_dict"])
    except Exception as ex:
        print("CHECKPOINT LOADING EXCEPTION: " + traceback.format_exc())


def check_best_model_of_epoch(
    tc,
    current_epoch,
):
    """This function checks if the current model is the best in the
    epoch until now. In such a case, the model is stored and every
    variable referencing the best losses and accuracies until now
    are updated with those of the current model.

    Args:
        tc (training_info_container) : container of every info related to
        the current training.
        current_epoch (int): tells the current epoch of training.

    """
    if tc.current_model_valid_loss < tc.current_epoch_best_valid_loss:
        save_path = "pruning_best_epoch_" + str(current_epoch) + "_model.pt"
        store_model_and_optim_state(
            tc=tc,
            path=save_path,
        )
        tc.current_epoch_best_valid_loss = tc.current_model_valid_loss
        tc.current_epoch_best_valid_acc = tc.current_model_valid_acc
        tc.current_epoch_best_train_acc = tc.current_model_train_acc
        tc.current_epoch_best_train_loss = tc.current_model_train_loss
        tc.current_epoch_best_lr = tc.current_model_lr


def early_stop_check(
    epoch,
    tc,
    early_stop,
):
    """This function checks whether the early stop is to be applied,
    and updates the variables that will allow it to be checked in any
    future epochs.

    Args:
        tc (training_info_container): container of every info related to
        the current training.
        epoch (int): actual completed epoch.
        early_stop (int): early stop threshold.

    Returns:
        bool: False if early stop must not be enforced, True otherwise.
    """
    if tc.current_epoch_best_valid_loss < tc.global_best_valid_loss:
        store_model_and_optim_state(tc=tc, path="pruning_best_global_model.pt")
        tc.stop_count = 0
        tc.global_best_valid_loss = tc.current_epoch_best_valid_loss
        tc.global_best_train_loss = tc.current_epoch_best_train_loss
        tc.global_best_valid_acc = tc.current_epoch_best_valid_acc
        tc.global_best_train_acc = tc.current_epoch_best_train_acc
        tc.best_epoch = epoch
        return False
    else:
        tc.stop_count += 1
        if tc.stop_count >= early_stop:
            message = "".join(
                [
                    "EARLY STOPPIING:\n Total epochs: \n",
                    str(epoch),
                    " \n   Best epoch: ",
                    str(tc.best_epoch),
                    " \n   with Validation Loss: ",
                    str(round(tc.global_best_valid_loss, 2)),
                    " \n   with Validation Accuracy: ",
                    str(round(100 * tc.global_best_valid_acc, 2)),
                ]
            )
            print(message)
            send_telegram_message(message=message)
            return True  # EARLY STOPPING
        return False


def get_new_report_filename():
    """This function returns a new filename which can be used
    to store the information related to the current training.
    In order to make each filename different and easily accessable,
    each filename is concatenated with the timestamp of the
    training start.

    Returns:
        string: new filename.
    """
    from datetime import datetime

    return "".join(
        [
            "TRAINING_RESULTS_",
            str(datetime.now().strftime("_%Y-%m-%d__%H-%M-%S")),
            ".txt",
        ]
    )


def readapt_smart_epoch_leap(current_learning_rate, current_smart_epoch_leap):
    """This is a simple function which readapts the smart leap tollerance if
    if learning rate has become too small.

    Args:
        current_learning_rate (float): learning rate of current model.
        current_smart_epoch_leap (int): current smart leap tollerance.

    Returns:
        int: new smart leap
    """
    if current_learning_rate < 0.0001:
        return current_smart_epoch_leap * 2
    return current_smart_epoch_leap


def terminate_training(
    tc,
):
    """This function handles the termination of the pruning based training
    thugh the following steps:
    1) The best model is loaded from the pytorch checkpoint.
    2) Print a message and send it through telegram, telling the most
       important info about the completed training (best epoch,...).
    3) Delete any intermidiate pytorch checkpoint to free some memory.
    4) Return the best model along with the history as dataframe and
       a dictionary with the most important performance parms of the model.


    Args:
        tc (training_info_container): container of every info related to
        the current training.

    Returns:
        tuple: best trained model, DataFrame table history, dictionary with relevant info
    """
    best_model, optimizer = load_model_and_optim_state("pruning_best_global_model.pt")
    best_model.current_optimizer = optimizer
    message = "".join(
        [
            "Best epoch: ",
            str(tc.best_epoch),
            " with loss: ",
            str(round(tc.global_best_valid_loss, 2)),
            " and acc: ",
            str(round(100 * tc.global_best_valid_acc, 2)),
            "%",
        ]
    )
    send_telegram_message(message=message)
    print(message)
    tc.history = DataFrame(
        tc.history, columns=["train_loss", "valid_loss", "train_acc", "valid_acc"]
    )
    for file_names in listdir("../code/"):
        if "pruning_best_epoch_" in file_names:
            remove("../code/" + file_names)
    return (
        best_model,
        tc.history,
        {
            "best_val_loss": tc.global_best_valid_loss,
            "best_epoch": tc.best_epoch,
            "best_val_acc": tc.global_best_valid_acc,
        },
    )


def optim_lr_update(tc, gamma_range, current_model_index):
    """This function performs an updated over the current optimizer to change
    the lr according to the related gamma. This function allows to specify a
    different lr for each optimizer of the current epoch, exploring different
    branches of the tree of the possible models.

    Args:
        tc (training_info_container): container of every info related to
        the current training.
        gamma_range (list of float): gamma range from which extract the gamma needed
        to update the lr.
        current_model_index (int): index of the current model useful to access the
        correct gamma.

    Returns:
        float: the new learning rate of the optimizer of the current model.
    """
    new_lr = (
        tc.current_optimizer.param_groups[0]["lr"] * gamma_range[current_model_index]
    )
    for param_group in tc.current_optimizer.param_groups:
        param_group["lr"] = new_lr
    tc.current_optimizer = optim.SGD(
        tc.current_model.parameters(),
        lr=new_lr,
        momentum=tc.current_optimizer.state_dict()["param_groups"][0]["momentum"],
        weight_decay=tc.current_optimizer.state_dict()["param_groups"][0][
            "weight_decay"
        ],
    )
    return new_lr


def optim_lr_update_with_precomputed_lrs(tc, precomputed_lrs, current_model_index):
    """Similar to the above function, this version makes use of the given
    precomputed lrs as new lrs.

    Args:
        tc (training_info_container): container of every info related to
        the current training.
        gamma_range (list of float): gamma range from which extract the gamma needed
        to update the lr.
        current_model_index (int): index of the current model useful to access the
        correct gamma.

    Returns:
        float: the new learning rate of the optimizer of the current model.
    """
    new_lr = precomputed_lrs[current_model_index]
    for param_group in tc.current_optimizer.param_groups:
        param_group["lr"] = new_lr
    tc.current_optimizer = optim.SGD(
        tc.current_model.parameters(),
        lr=new_lr,
        momentum=0.9,
        weight_decay=tc.current_optimizer.state_dict()["param_groups"][0][
            "weight_decay"
        ],
    )
    return new_lr


def pruning_based_training(
    original_model,
    criterion,
    original_optimizer,
    train_loader,
    val_loader,
    gamma_range,
    early_stop=5,
    n_epochs=20,
    smart_epoch_leap=10000,
    is_dynamic_leap=False,
    perform_cooldown=False,
):
    """This function handles the training of the given model using some sub techniques
        used to improve validation loss and generalisation:

            * Early stopping: if after "early_stop" epochs there is not improvement in the
              validation loss, the function stops and returns the model with the best loss.
              This techinque avoid the overfitting introduced by the training after a number
              of epochs where the validation loss starts to increase and the training loss
              keeps decreasing, showing that the model is performing better on the training
              sample but worse on the validation ones.

            * Backtracking with Constraint propagation: I chose to use the starting LR as a
              starting point for multiple models per-epoch: given the gammas array containing
              several float values, during each epoch the function will train as much models
              as there are gamma values, using for each of them the LR of the best model trained
              in the previous epoch (even if worse than the best until that moment) times the
              related gamma, and for each epoch I will pick the best model of such epoch
              (i.e., lowest Valid loss), in order to use its LR a starting LR for the next epoch.
              This technique proves very useful because it allows to follow the validation loss
              very efficiently even on models with several different configurations of hyperparameters,
              getting a first idea of how the learning rate should be scheduled epoch by epoch,
              and also a possible idea of one of the best accuracies achievable by the model.

                Intelligent leap
                * Smart skip (Defined by me): Considering each model in each epoch as a starting
                point of several possible models in the next epoch, this set of models represents
                a tree in which each model is a branch. The algorithm already applies a kind of
                pruning approach, when each model of an epoch is discarded except for the best one,
                but here I use some heuristics gathered during my tests: in a wide range of intervals,
                only a few, consecutive, of them will return useful results, and the subsequent
                ones will simply be useless, so to avoid unnecessarily calculating models that with
                a high probability will be worse than those already calculated, I devised a smart_epoch_leap
                variable, the concept of which is similar to that of early_stop: after a given number
                of models in the same epoch do not produce better results than the previous ones in
                the same epoch, the function skips all subsequent models. This technique is very useful
                for speeding up the process and allowing a finer search over a range of values, and
                the value of smart_epoch_leap should be chosen according to the size of the array_range:
                higher array sizes lead to more models with similar lr, and thus also to slower
                improvements, so the value of smart_epoch_leap should be higher to make the check faster.
                The value of smart_epoch_leap should be higher to make checking happen more infrequently,
                otherwise many patterns may be eliminated prematurely. on the other hand, smaller
                array sizes require a lower skip size, because the variations in the lr will be larger
                and have a more distinct effect on the patterns, so the tolerance can be decreased
                accordingly. The is_dynamic_leap argument allows the tolerance to be dynamically increased
                according to the size of the learning rate (smaller lr -> increased tolerance), thus
                avoiding premature pruning with small changes in the learning rate. However, this
                technique should not be used on the first run, because the model may show good losses
                in one lr window, only to show better results in another lr window away from the first,
                so it is advisable to run at least once without this technique for the first time, in
                order to specify finer-grained lr changes with a smart_epoch_leap in next executions, focusing
                the exploration on the best candidate windows

    Translated with www.DeepL.com/Translator (free version)
        Args:
            original_model (pytorch model): model to train
            criterion (pytorch criterion): criterion used to produce the loss
            original_optimizer (pytorch optim): optimizer related to the model
            train_loader (pytorch train loader): loader used to fetch the data for training
            val_loader (pytorch val loader): loader used to fetch the data for validation
            gamma_range (list/array of float/int): gamma values used to multiply the learning rates,
            obtaining as much models as gammas.
            early_stop (int, optional): Epochs without improvement after which early stop
            will be enforced. Defaults to 5.
            n_epochs (int, optional): total number of epochs given for the training, after which the
            training will stop independently of its results. Defaults to 20.
            smart_epoch_leap (int, optional): number of models without improvement in an epoch after
            which the model will skip the next models jumping to the next epoch. Defaults to 10000.
            is_dynamic_leap (bool, optional): this variables tells if the smart_epoch_leap values should be
            adapted to the smaller learning rates. Default to True.
            perform_cooldown (bool, optional): this variables tells to the training function to sleep each time
            an epoch ends to cooldown the hardware in case of very long runs.

        Returns:
            tuple: model, history, dictionary of model's statistics.
    """
    tc = training_info_container()
    tc.stop_count = 0
    tc.history = []
    original_model.epochs = 0
    num_of_models_per_epoch = len(gamma_range)
    # all_models_message = ""
    tc.scaler = None
    tc.global_best_train_loss = sys.float_info.max
    tc.global_best_valid_loss = sys.float_info.max
    tc.global_best_train_acc = 0
    tc.global_best_valid_acc = 0
    tc.best_epoch = 0
    tc.report_file_name = get_new_report_filename()
    train_loader_len = len(train_loader.dataset)
    valid_loader_len = len(val_loader.dataset)
    is_smart_leap_updated = False
    with open(tc.report_file_name, "w") as f:
        f.write(
            "#_EPOCH , #_Model :, Training Loss, Validation Loss, Train Accuracy %, Valid Accuracy %, Learning Rate"
        )
    for epoch in range(n_epochs):
        tc.current_epoch_best_train_loss = sys.float_info.max
        tc.current_epoch_best_valid_loss = sys.float_info.max
        tc.current_epoch_best_train_acc = 0
        tc.current_epoch_best_valid_acc = 0
        tc.current_epoch_best_lr = 0
        smart_epoch_leap_no_improvement_counter = 0
        for act_model_index in log_progress(
            range(0, num_of_models_per_epoch),
            every=1,
            name="EPOCH -" + str(epoch) + "- completed models: ",
        ):
            tc.scaler = None  # Needed to avoid using the same scaler across models
            if epoch != 0:
                load_model_and_optim_state_with_scaler(
                    tc=tc, path="pruning_best_epoch_" + str(epoch - 1) + "_model.pt"
                )
            else:
                tc.current_model = deepcopy(original_model)
                tc.current_optimizer = deepcopy(original_optimizer)
            tc.current_model_train_loss = 0
            tc.current_model_valid_loss = 0
            tc.current_model_train_acc = 0
            tc.current_model_valid_acc = 0
            tc.current_model.epochs = epoch
            tc.current_model_lr = optim_lr_update(
                tc=tc,
                gamma_range=gamma_range,
                current_model_index=act_model_index,
            )
            tc.current_model.train()
            loaded_batches = 0
            tc.scaler = cuda.amp.GradScaler() if tc.scaler == None else tc.scaler
            for data, label in log_progress(
                sequence=train_loader, every=1, name="Used batches: "
            ):
                loaded_batches += 1
                data, label = data.cuda(), label.cuda()
                tc.current_optimizer.zero_grad(set_to_none=True)
                with cuda.amp.autocast():
                    output = tc.current_model(data)
                    loss = criterion(output, label)
                tc.scaler.scale(loss).backward()
                tc.scaler.step(tc.current_optimizer)
                tc.scaler.update()
                tc.current_model_train_loss += loss.item() * data.size(0)
                _, pred = max(output, dim=1)
                tc.current_model_train_acc += mean(
                    pred.eq(label.data.view_as(pred)).type(FloatTensor)
                ).item() * data.size(0)
            tc.current_model.epochs += 1
            with no_grad():
                tc.current_model.eval()
                for data, label in val_loader:
                    data, label = data.cuda(), label.cuda()
                    with cuda.amp.autocast():
                        output = tc.current_model(data)
                        loss = criterion(output, label)
                    tc.current_model_valid_loss += loss.item() * data.size(0)
                    _, pred = max(output, dim=1)
                    tc.current_model_valid_acc += mean(
                        pred.eq(label.data.view_as(pred)).type(FloatTensor)
                    ).item() * data.size(0)

                tc.current_model_train_loss /= train_loader_len
                tc.current_model_valid_loss /= valid_loader_len
                tc.current_model_train_acc /= train_loader_len
                tc.current_model_valid_acc /= valid_loader_len

                tc.history.append(
                    [
                        tc.current_model_train_loss,
                        tc.current_model_valid_loss,
                        tc.current_model_train_acc,
                        tc.current_model_valid_acc,
                    ]
                )

                current_epoch_message = (
                    f"\nEPOCH #: {epoch}, Model #: {act_model_index}, Training Loss: {tc.current_model_train_loss:.4f}, Validation Loss: {tc.current_model_valid_loss:.4f}"
                    f", Training Accuracy: {100 * tc.current_model_train_acc:.2f}%, Validation Accuracy: {100 * tc.current_model_valid_acc:.2f}%, Learning Rate: {tc.current_model_lr:.9f}"
                )
                print(current_epoch_message)
                with open(tc.report_file_name, "a") as f:
                    f.write(current_epoch_message)

                # SMART SKIP CHECK
                if is_dynamic_leap and (not is_smart_leap_updated):
                    smart_epoch_leap = readapt_smart_epoch_leap(
                        tc.current_model_lr, current_smart_epoch_leap=smart_epoch_leap
                    )
                if tc.current_epoch_best_valid_loss < tc.current_model_valid_loss:
                    smart_epoch_leap_no_improvement_counter += 1
                    if smart_epoch_leap_no_improvement_counter >= smart_epoch_leap:
                        send_telegram_message(
                            f"EPOCH SKIP: No improvement after {smart_epoch_leap} models"
                        )
                        break
                else:
                    smart_epoch_leap_no_improvement_counter = 0

            check_best_model_of_epoch(tc=tc, current_epoch=epoch)
        (
            tc.best_model_of_epoch,
            tc.related_optimizer,
        ) = load_model_and_optim_state("pruning_best_epoch_" + str(epoch) + "_model.pt")
        message = (
            f"PARAMS OF BEST MODEL IN THE EPOCH:\n\n   EPOCH #: {epoch} \n   Training Loss: "
            f"{tc.current_epoch_best_train_loss:.4f} \n   Validation Loss: {tc.current_epoch_best_valid_loss:.4f}"
            f" \n   Training Accuracy: {100 * tc.current_epoch_best_train_acc:.2f}% \n   Validation Accuracy: "
            f"{100 * tc.current_epoch_best_valid_acc:.2f}% \n   Learning Rate: {tc.current_epoch_best_lr:.9f}"
        )
        send_telegram_message(message=message)
        should_early_stop = early_stop_check(
            tc=tc,
            epoch=epoch,
            early_stop=early_stop,
        )
        if should_early_stop:
            return terminate_training(tc=tc)
        if perform_cooldown:
            from time import sleep

            print("--ENFORCING SLEEP FOR THE REST OF THE HARDWARE--")
            sleep(120.0)
            print("--COOLDOWN COMPLETED--")
    return terminate_training(tc=tc)


from numpy import arange


def pruning_based_training_with_precomputed_lrs(
    original_model,
    criterion,
    original_optimizer,
    train_loader,
    val_loader,
    precomputed_lrs,
    base_gammas=arange(0.05, 1.05, 0.5),
    early_stop=5,
    n_epochs=20,
):
    tc = training_info_container()
    tc.stop_count = 0
    tc.history = []
    original_model.epochs = 0
    # all_models_message = ""
    tc.scaler = None
    tc.global_best_train_loss = sys.float_info.max
    tc.global_best_valid_loss = sys.float_info.max
    tc.global_best_train_acc = 0
    tc.global_best_valid_acc = 0
    tc.best_epoch = 0
    tc.report_file_name = get_new_report_filename()
    train_loader_len = len(train_loader.dataset)
    valid_loader_len = len(val_loader.dataset)
    number_of_precomputed_epochs = len(precomputed_lrs)
    with open(tc.report_file_name, "w") as f:
        f.write(
            "#_EPOCH , #_Model :, Training Loss, Validation Loss, Train Accuracy %, Valid Accuracy %, Learning Rate"
        )
    # print(precomputed_lrs)
    for epoch in range(n_epochs):
        tc.current_epoch_best_train_loss = sys.float_info.max
        tc.current_epoch_best_valid_loss = sys.float_info.max
        tc.current_epoch_best_train_acc = 0
        tc.current_epoch_best_valid_acc = 0
        tc.current_epoch_best_lr = 0

        current_epoch_gammas = (
            precomputed_lrs[epoch]
            if epoch < number_of_precomputed_epochs
            else base_gammas
        )
        num_of_precomputed_lrs = len(current_epoch_gammas)

        for act_model_index in log_progress(
            range(0, num_of_precomputed_lrs),
            every=1,
            name="EPOCH -" + str(epoch) + "- completed models: ",
        ):
            tc.scaler = None  # Needed to avoid using the same scaler across models
            if epoch != 0:
                load_model_and_optim_state_with_scaler(
                    tc=tc, path="pruning_best_epoch_" + str(epoch - 1) + "_model.pt"
                )
            else:
                tc.current_model = deepcopy(original_model)
                tc.current_optimizer = deepcopy(original_optimizer)

            current_epoch_learning_rates = (
                (current_epoch_gammas)
                if epoch < number_of_precomputed_epochs
                else current_epoch_gammas * tc.current_optimizer.param_groups[0]["lr"]
            )
            print(current_epoch_learning_rates)

            tc.current_model_train_loss = 0
            tc.current_model_valid_loss = 0
            tc.current_model_train_acc = 0
            tc.current_model_valid_acc = 0
            tc.current_model.epochs = epoch
            tc.current_model_lr = optim_lr_update_with_precomputed_lrs(
                tc=tc,
                precomputed_lrs=current_epoch_learning_rates,
                current_model_index=act_model_index,
            )
            tc.current_model.train()
            loaded_batches = 0
            tc.scaler = cuda.amp.GradScaler() if tc.scaler == None else tc.scaler

            for data, label in log_progress(
                sequence=train_loader, every=1, name="Used batches: "
            ):
                loaded_batches += 1
                data, label = data.cuda(), label.cuda()
                tc.current_optimizer.zero_grad(set_to_none=True)
                with cuda.amp.autocast():
                    output = tc.current_model(data)
                    loss = criterion(output, label)
                tc.scaler.scale(loss).backward()
                tc.scaler.step(tc.current_optimizer)
                tc.scaler.update()
                tc.current_model_train_loss += loss.item() * data.size(0)
                _, pred = max(output, dim=1)
                tc.current_model_train_acc += mean(
                    pred.eq(label.data.view_as(pred)).type(FloatTensor)
                ).item() * data.size(0)
            tc.current_model.epochs += 1
            with no_grad():
                tc.current_model.eval()
                for data, label in val_loader:
                    data, label = data.cuda(), label.cuda()
                    with cuda.amp.autocast():
                        output = tc.current_model(data)
                        loss = criterion(output, label)
                    tc.current_model_valid_loss += loss.item() * data.size(0)
                    _, pred = max(output, dim=1)
                    tc.current_model_valid_acc += mean(
                        pred.eq(label.data.view_as(pred)).type(FloatTensor)
                    ).item() * data.size(0)

                tc.current_model_train_loss /= train_loader_len
                tc.current_model_valid_loss /= valid_loader_len
                tc.current_model_train_acc /= train_loader_len
                tc.current_model_valid_acc /= valid_loader_len

                tc.history.append(
                    [
                        tc.current_model_train_loss,
                        tc.current_model_valid_loss,
                        tc.current_model_train_acc,
                        tc.current_model_valid_acc,
                    ]
                )

                current_epoch_message = (
                    f"\nEPOCH #: {epoch}, Model #: {act_model_index}, Training Loss: {tc.current_model_train_loss:.4f}, Validation Loss: {tc.current_model_valid_loss:.4f}"
                    f", Training Accuracy: {100 * tc.current_model_train_acc:.2f}%, Validation Accuracy: {100 * tc.current_model_valid_acc:.2f}%, Learning Rate: {tc.current_model_lr:.9f}"
                )
                print(current_epoch_message)
                with open(tc.report_file_name, "a") as f:
                    f.write(current_epoch_message)
                # NO SMART SKIP CHECK

            check_best_model_of_epoch(tc=tc, current_epoch=epoch)
        (
            tc.best_model_of_epoch,
            tc.related_optimizer,
        ) = load_model_and_optim_state("pruning_best_epoch_" + str(epoch) + "_model.pt")
        message = (
            f"PARAMS OF BEST MODEL IN THE EPOCH:\n\n   EPOCH #: {epoch} \n   Training Loss: "
            f"{tc.current_epoch_best_train_loss:.4f} \n   Validation Loss: {tc.current_epoch_best_valid_loss:.4f}"
            f" \n   Training Accuracy: {100 * tc.current_epoch_best_train_acc:.2f}% \n   Validation Accuracy: "
            f"{100 * tc.current_epoch_best_valid_acc:.2f}% \n   Learning Rate: {tc.current_epoch_best_lr:.9f}"
        )
        send_telegram_message(message=message)
        should_early_stop = early_stop_check(
            tc=tc,
            epoch=epoch,
            early_stop=early_stop,
        )
        if should_early_stop:
            return terminate_training(tc=tc)
    return terminate_training(tc=tc)
