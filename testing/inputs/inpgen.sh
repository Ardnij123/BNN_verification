#!/bin/bash

for l in 25 100 400; do
    for pos in $(eval echo {0..$(( $l-1 ))}); do
        echo -n "$pos "
    done | head --bytes=-1 > "inpbits_0_${l}_0.txt"
    for free in 8 16 24; do
        cat "inpbits_0_${l}_$(( $free-8 )).txt" | cut -d' ' -f'9-' > "inpbits_0_${l}_${free}.txt"
    done
done
