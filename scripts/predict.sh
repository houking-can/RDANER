# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/yhj/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/yhj/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/yhj/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/yhj/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate torch1.3
