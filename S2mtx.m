function S2 = S2mtx(n)
S2 = zeros(n^2,n*(n+1)/2);
stx = 1;
sty = 1;
for I = 1:n
    for J = 1:I
        L = n+1-J;
        LIJ = zeros(n,L);
        if I==J
            LIJ(n-L+1:n,:) = eye(L);
        end
        if I > J
            LIJ(J,I-J+1)=1;
        end
        S2(stx:stx+n-1,sty:sty+L-1)=LIJ;
        sty = sty+L;
    end
    sty = 1;
    stx = stx + n;
end
end
        