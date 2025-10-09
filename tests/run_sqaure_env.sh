for i in {1..20}; do
    # 定义一个唯一的会话名称
    SESSION_NAME="worker_$i"
    
    # 1. 创建一个新的 tmux 会话，不进入该会话 (-d)
    #    -s: 指定会话名称
    tmux new-session -d -s "$SESSION_NAME" 
    
    # 2. 向该会话发送命令 (send-keys)
    #    C-m: 相当于按下回车键执行命令
    # conda activate vlarl_infra
    tmux send-keys -t "$SESSION_NAME" "conda activate vlarl_infra" C-m
    tmux send-keys -t "$SESSION_NAME" "poetry run vlarl-run-worker robomimic-v1 --num-episodes 100000 --env.name square-img --max-episode-steps 400" C-m
    
    echo "Started worker $i in tmux session: $SESSION_NAME"
done