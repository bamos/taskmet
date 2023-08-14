#!/bin/bash

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

for i in $(seq 0 $((${#combinations[@]} - 1)))
do
  echo "$i: ${combinations[$i]}"
done

i=1
combination=(${combinations[$i]//,/ })

lr="${combination[0]}"
weight_decay="${combination[1]}"
iters="${combination[2]}"
activation="${combination[3]}"
dim_distract="${combination[4]}"

echo "weight_decay: $weight_decay"
echo "lr: $lr"
echo "iters: $iters"
echo "activation: $activation"
echo "dim_distract: $dim_distract"

