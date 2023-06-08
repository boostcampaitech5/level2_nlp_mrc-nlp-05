
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

alias cit='cd /opt/ml/input/code/install' # 1번
alias cn='conda create -n mrc python=3.10 -y' # 2번
alias cm='chmod +x install_requirements.sh' # 3번
alias it='bash ./install_requirements.sh' # 4번
alias t='python input/code/train.py'
alias i='python input/code/inference.py'
alias tn='tmux new -s 0'
alias tmux_n='tmux new -s 0'
alias ta='tmux attach -t 0'
alias tmux_a='tmux attach -t 0'
alias s='source .bashrc'
alias c='conda activate mrc'
alias cl='conda list'
alias g='git'
alias gs='git status'
alias ga='git add .'
alias gc='git commit'
alias gcm='git commit -m'
alias gp='git push origin'