#!/bin/bash
# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_ARRAY_TASK_ID	

# Parse arguments
declare -a args
declare -a values
for arg in "$@"
do
    if [[ "$arg" == *=* ]]; then
        # Argument is a value
        value=${arg#*=}
        values+=("$value")
    else
        # Argument is a flag
        if [[ ${#values[@]} -ne 0 ]]; then
            args+=("${values[@]}")
            values=()
        fi
        args+=("$arg")
    fi
done
if [[ ${#values[@]} -ne 0 ]]; then
    args+=("${values[@]}")
fi

# Loop over all combinations
function cartesian_product {
  # Adapted from https://stackoverflow.com/a/1782956
  local result=()
  local params=("${!1}")
  local lengths=()
  for param in "${params[@]}"
  do
    lengths+=($(echo "$param" | grep -o ',' | wc -l | awk '{print $1 + 1}'))
  done
  local max_index=$(( ${#params[@]} - 1 ))
  local indices=()
  for i in $(seq 0 $max_index); do indices[$i]=0; done
  local done=false
  while [[ $done == false ]]
  do
    local tuple=()
    for i in $(seq 0 $max_index)
    do
      local param="${params[$i]}"
      local index="${indices[$i]}"
      local value=$(echo "$param" | cut -d',' -f$(( $index + 1 )) --output-delimiter='')
      tuple+=("$value")
    done
    result+=("$(IFS=','; echo "${tuple[*]}")")
    local i=$max_index
    while [[ $i -ge 0 ]]
    do
      if [[ ${indices[$i]} -lt $((${lengths[$i]} - 1)) ]]
      then
        indices[$i]=$((${indices[$i]} + 1))
        for j in $(seq $(($i + 1)) $max_index)
        do
          indices[$j]=0
        done
        break
      fi
      ((i--))
    done
    if [[ $i -lt 0 ]]
    then
      done=true
    fi
  done
  echo "${result[@]}"
}

combinations=($(cartesian_product args[@]))

export XLA_PYTHON_CLIENT_MEM_FRACTION=.45

combination=(${combinations[$SLURM_ARRAY_TASK_ID]//,/ })

activation="${combination[0]}"
seed="${combination[1]}"
sizes="${combination[2]}"
regularization_coeff="${combination[3]}"


agent="metric"
exp=model_size_diag

IFS='-' read -ra ADDR <<< "$sizes"

for size in "${ADDR[@]}"; do
  outdir="exp/$exp/$agent/hidden_size=$size,activation=$activation,regularization_coeff=$regularization_coeff/$seed"
  mkdir -p $outdir
  echo "Running $agent with arguments hidden_size=$size,activation=$activation"
  python main.py with_inv_jac_model=True metric_conditional=True full_network=True diag_metric=True seed=$seed agent_type=$agent regularization_coeff=$regularization_coeff exp=$exp metric_activation=$activation out_dir=$outdir model_hidden_dim=$size > $outdir/log.txt &
done
wait