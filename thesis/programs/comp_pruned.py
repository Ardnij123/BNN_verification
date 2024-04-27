num_literals = 20

with open('comp_count_pruned.aspif', 'w') as f:
    f.write('asp 1 0 0\n')

    # write choice for literals
    for i in range(num_literals):
        f.write(f"1 1 1 {i+1} 0 0\n")

    nl = num_literals + 1
    # write ifs
    for i in range(num_literals):
        f.write(f"1 0 1 {nl} 1 {i+1} {num_literals}")
        for j in range(num_literals):
            f.write(f" {j+1} 1")
        f.write('\n')

        if i == 0:
            f.write(f"1 0 1 {nl+1} 0 1 -{nl}\n")
        else:
            f.write(f"1 0 1 {nl+1} 0 2 {nl-2} -{nl}\n")
        nl += 2
    f.write(f"1 0 1 {nl} 0 1 {nl-2}\n")

    # show statements
    for i in range(num_literals):
        f.write(f"4 {4 if i<9 else 5} a({i+1}) 1 {i+1}\n")
    for i in range(num_literals):
        f.write(f"4 {4 if i<10 else 5} n({i}) 1 {num_literals + 2*(i+1)}\n")
    f.write(f"4 {4 if num_literals<10 else 5} n({num_literals}) 1 {num_literals + 2*(num_literals) + 1}\n")

    # end of file
    f.write("0")
