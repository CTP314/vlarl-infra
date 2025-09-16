# 🚀 VLARL-Infra

**VLARL-Infra** (Vision Language Action Reinforcement Learning Infrastructure) is a foundational framework for building and running distributed reinforcement learning agents. It's designed to streamline the communication between environment workers and a centralized training server, enabling efficient and scalable RL experiments.

## ✨ Features

  * **Distributed Communication**: Utilizes `websockets` and `msgpack` for fast, secure, and asynchronous data transfer between the central server and multiple environment workers.
  * **Modular Design**: Separates the core infrastructure (`vlarl-infra`) from the client-side components (`vlarl-client`), allowing for independent development and deployment.
  * **Command-Line Interface**: Provides a user-friendly CLI powered by `tyro` for starting and managing RL tasks.
  * **Gymnasium Integration**: Seamlessly works with `Gymnasium` environments for standardized and flexible environment interaction.

-----

## 🛠️ Installation

The easiest way to get started is by cloning the repository and using Poetry to handle the dependencies.

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:CTP314/vlarl-infra.git
    cd vlarl-infra
    ```

2.  **Install dependencies with Pip:**

    ```bash
    pip install -e .
    ```

-----

### 🚀 Usage

`vlarl-infra` comes with a command-line utility to launch environment workers.

#### Running a Worker

To start a worker that connects to the central RL server, use the `vlarl-run-worker` command.

```bash
vlarl-run-worker
```

examples:

```
vlarl-run-worker dummy-v1 --log-level DEBUG --num-episodes 1000
```

This command will launch an environment worker that interacts with a specified environment and sends experience data to the central server for training.