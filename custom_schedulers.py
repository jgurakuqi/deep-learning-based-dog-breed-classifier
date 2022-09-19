from numpy import arange, concatenate, array


class custom_cyclic_scheduler:
    __name__ = "custom_cyclic_scheduler"

    def __init__(this, init_lr, max_lr, step_size_up=0, step_size=1, reverse=False):
        """This scheduler replicates the behaviour of a cyclic LR Scheduler, but
        introducing the step_size feature typical of other schedulers, which
        delays the update of the lr of how many steps we want.

        Args:
            this (custom_cyclic_scheduler): Reference to the instance.
            init_lr (float): Starting lr
            max_lr (float): Maximum lr (included)
            step_size_up (int, optional): How many steps are needed to ascend and descend from a lr spike. Defaults to 0.
            step_size (int, optional): How many steps the scheduler will wait before of changing the lr. Defaults to 1.
            reverse (bool, optional): Tells if the scheduler must start the cyclic behaviour from the maxlr or the minlr. Defaults to False.

        Raises:
            Exception: Checks if the max and init lr are both positive.
            Exception: Checks if the max lr is bigger than the init lr.
            Exception: Checks if the number of steps needed to reach the top/bottom of a spike is bigger than or equal to one.
            Exception: Checks if the number of steps needed to update the lr is bigger than or equal to 0
        """
        step_size = int(step_size)
        step_size_up = int(step_size_up)
        if init_lr <= 0 or max_lr <= 0:
            raise Exception("Only positive learning rates are permitted!")
        if max_lr <= init_lr:
            raise Exception(
                "The maximum learning rate must be higher than the minimum one!"
            )
        if step_size_up < 1:
            raise Exception(
                "Only step-up sizes greater than or equal to 1 are permitted!"
            )
        if step_size < 0:
            raise Exception("Only step sizes greater than or equal to 0 are permitted!")
        this.__actual_epoch = 0
        # this.__step_size = step_size + 1
        this.__step_size = step_size + 1
        this.__all_lrs = concatenate(
            (arange(init_lr, max_lr, (max_lr - init_lr) / step_size_up), [max_lr])
        )
        print(this.__all_lrs)
        this.__actual_lr_index = len(this.__all_lrs) - 1 if reverse else 0
        # this.__epochs_before_step = step_size
        this.__lr_index_step = -1 if reverse else 1

    def step(this, optimizer):
        """This is the method which will handle the update/step operation of this custom scheduler.
        The scheduler.step, as stated by nowadays pytorch standards, needs to be invoked at the end
        of the running epoch, so this function will check if the next epoch matches the epoch at
        which it should perform the lr update, and if so it will update the lr using the numpy array
        of learning rates computed in the init method. Each time that the scheduler will have used
        all the lrs into one direction of the lrs' array, it will change the direction in which it
        will pick the lr from the array.

        Args:
            this (custom_cyclic_scheduler): Reference to the instance.
            optimizer (torch.optim object): This is the optimiser to which the method will change the lrs.

        Returns:
            torch.optim: optimizer
        """
        # I chose + 1 here instead of doing it in the init otherwise the initial Lr would have been
        # used 1 time less than any other lr.
        if (this.__actual_epoch + 1) % this.__step_size == 0:
            this.__actual_lr_index += this.__lr_index_step
            next_index = this.__actual_lr_index + this.__lr_index_step
            # if next_index == 0 or next_index == (len(this.__all_lrs) - 1):
            if next_index == -1 or next_index == (len(this.__all_lrs)):
                this.__lr_index_step = -this.__lr_index_step

            new_lr = this.__all_lrs[this.__actual_lr_index]

            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        this.__actual_epoch += 1
        return optimizer


class custom_step_scheduler:
    __name__ = "custom_step_scheduler"

    def __init__(
        this,
        optimizer,
        gamma=0.5,
        step_sizes=[],
        default_step_size=1,
        learning_rate_decay=0.0,
        threshold=0.00001,
    ):
        """This scheduler replicates the behaviour of a step LR Scheduler, but
        introducing the cyclical step_sizes feature, which allows to the different
        learning rate to perdure for a different amount of time.

        Args:
            this (custom_step_scheduler): Reference to the instance.

            optimizer (torch.optim object): This is the pytorch optimizer.
            gamma (float): The proportion by which the lr will be decreased. Default to 0.5.
            step_sizes (list/array of int, optional): How many steps the scheduler will wait before of changing the lr.
            default_step_size (int, optional): Tells which step_size to resume after consuming the step_sizes list. Default to 1.
            learning_rate_decay (float, optional): It's the amount of learning rate decay used on each step. Default to 0.0.
            threshold (float, optional): It's the minimum possible lr that the scheduler can compute. If such threshold is
            violated, than the scheduler won't decrease the lr anymore, and will start to use the last previous smallest
            lr before of the threshold. Default to 0.00001.

        Raises:
            Exception: Checks if the starting learning rates of the given optimizer are positive.
            Exception: Checks if the number of steps after the depletion of step_sizes is at least one.
            Exception: Checks if the gamma used to update the learning rate is included in the 0-1 interval.
            Exception: Checks if each of the given step sizes is at least 0.
            Exception: Checks if the learning rate decay is positive.
            Exception: Checks if the threshold is positive.
        """
        default_step_size = int(default_step_size)
        step_sizes = array(step_sizes, dtype=int)
        gamma = float(gamma)
        for param_group in optimizer.param_groups:
            if param_group["lr"] <= 0.0:
                raise Exception("Only positive learning rates are permitted!")
            this.actual_lr = param_group["lr"]
        if default_step_size < 1:
            raise Exception("Only (default) step sizes greater than 1 are permitted!")
        if gamma <= 0.0 or gamma >= 1.0:
            raise Exception(
                "Only alpha rates included smaller than 1 and bigger than 0 are permitted!"
            )
        for step_size in step_sizes:
            if step_size < 0:
                raise Exception(
                    "Only step sizes greater than or equal to 0 are permitted!"
                )
        if learning_rate_decay < 0:
            raise Exception("Only positive learning_rate_decay is permitted!")
        if threshold <= 0:
            raise Exception("Only positive thresholds are permitted!")

        this.default_step_size = default_step_size
        this.step_sizes = step_sizes
        this.gamma = gamma
        this.default_step_size_counter = default_step_size
        this.consecutive_updates = 0
        this.learning_rate_decay = learning_rate_decay
        this.threshold = threshold
        print(this.learning_rate_decay)

    def is_step_update_available(this):
        """This method counts the number of times that the lr needs to be updated
        consequtively. If the step_sizes array is not empty, if:
            * The first valye is zero, the method will start to pop away every
              zero consequitvely places after this one, and for each of them will
              increment the consequtive_updates variable, until a non-zero number
              will be reached or the array will be depleted.
            * The first value is not zero, than it will be decreased turn by turn,
              until it become zero. The step invokation when such value becomes zero,
              will cause an increase of the variable consequtive_updates which keeps
              track of the number of updates to perform. Such value will be popped
              away, and if:
                * The next value is a number, nothing happens.
                * The next number is a zero, starts the same routine seen above with
                  sequences of zeros.

        Args:
            this (custom_step_scheduler): Reference to the instance.

        """
        if this.step_sizes.size > 0:
            if this.step_sizes[0] > 0:
                this.step_sizes[0] -= 1
            while this.step_sizes.size != 0 and this.step_sizes[0] == 0:
                _, this.step_sizes = this.step_sizes[0], this.step_sizes[1:]
                this.consecutive_updates += 1
            return
        else:
            this.default_step_size_counter -= 1
            if this.default_step_size_counter == 0:
                this.default_step_size_counter = this.default_step_size
                this.consecutive_updates += 1

    def step(this, optimizer):
        """This is the method which will handle the update/step operation of this custom scheduler.
        This method will first check the number of updates required by is_step_update_available,
        and then will update the learning rates as many times as consecutive_updates.

        Args:
            this (custom_step_scheduler): Reference to the instance.
            optimizer (torch.optim object): This is the optimiser to which the method will change the lrs.

        Returns:
            torch.optim: optimizer
        """
        if this.threshold < this.actual_lr:
            this.is_step_update_available()
            if this.consecutive_updates > 0:
                this.actual_lr = new_lr = this.actual_lr * pow(
                    this.gamma, this.consecutive_updates
                )
                print("PRE DECAY " + str(this.actual_lr))
                this.actual_lr -= this.actual_lr * this.learning_rate_decay
                print("AFTER DECAY " + str(this.actual_lr))

                this.consecutive_updates = 0
                if this.actual_lr < this.threshold:
                    this.actual_lr = (
                        this.threshold
                    )  # Threshold violation: lr limit reached
                    return optimizer
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr
        return optimizer
