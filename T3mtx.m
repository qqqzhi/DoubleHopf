function T3 = T3mtx(n)

T3 = zeros( n*(n+1)*(n+2)/6, n*n*(n+1)/2);
BgN = n*(n+1)/2;
T3(1:BgN,1:BgN) = eye(BgN);
stx = BgN + 1;
sty = BgN + 1;
for k = n-1:-1:1
    K2 = k*(k+1)/2;
    Bk = [zeros(K2,BgN-K2) eye(K2)];
    T3(stx:stx+K2-1, sty:sty+BgN-1) = Bk;
    stx = stx + K2;
    sty = sty + BgN;
end
