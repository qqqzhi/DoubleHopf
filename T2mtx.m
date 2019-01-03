function T2 = T2mtx(n)
T2 = zeros(n*(n+1)/2, n^2);
T2(1:n,1:n) = eye(n);
stx = n+1;
sty = n+1;
for k = (n-1):-1:1
    Bk = [zeros(k,n-k) eye(k)];
    T2(stx:stx+k-1,sty:sty+n-1) = Bk;
    stx = stx + k;
    sty = sty + n;
end
end