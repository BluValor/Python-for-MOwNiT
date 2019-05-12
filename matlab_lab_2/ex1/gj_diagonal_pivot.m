function gj_diagonal_pivot(A)

for m = 1:(size(A) - 1)
    for i = (m + 1):size(A)
        A(:, i) = A(:, i) + 10
    end
end