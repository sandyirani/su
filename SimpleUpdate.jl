

using TensorOperations
using LinearMaps

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


pd = 2
N = 5
D = 3
A = [zeros(1,1,1,1,pd) for j=1:N, for k = 1:N]
for j = 1:N
    for k = 1:N
        idx = (iseven(j+k)? 1, 2)
        A[j,k][1,1,1,1,idx] = 1
    end
end
SV = [eye(1) for j = 1:N-1, for k = 1:N]
SH = [eye(1) for j = 1:N, for k = 1:N-1]




#Global variables
sz = Float64[0.5 0; 0 -0.5]
sp = Float64[0 1; 0 0]
sm = sp'
#Htwosite = reshape(JK(sz,sz) + 0.5 * JK(sp,sm) + 0.5 * JK(sm,sp),2,2,2,2)
lambda = 3.0
sigZ = Float64[1 0; 0 -1]
sigX = Float64[0 1; 1 0]
Htwosite = reshape(JK(sigZ,sigZ) + lambda*0.25*JK(eye(2),sigX) + lambda*0.25*JK(sigX,eye(2))
# order for Htwosite is s1, s2, s1p, s2p



function mainLoop()
    numIter = 100
    tau = .2
    for iter = 1:numIter
        taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
        println("\n iteration = $iter")
        for j = 1:N
            for k = 1:N-1
                applyGateAndUpdateRight(taugate, true, j, k) #true = horiz
                applyGateAndUpdateDown(taugate, false, k, j) #false = vert
            end
        end
    end
end

function applyGateAndUpdateRight(g, row, col)

        merge(row,col,UP,false)
        merge(row,col,DOWN,false)
        merge(row,col,LEFT,false)
        merge(row,col,RIGHT,false)
        merge(row,col+1,UP,false)
        merge(row,col+1,RIGHT,false)
        merge(row,col+1,DOWN,false)
        Aleft = A[row,col]
        Aright = A[row,col+1]
        (Aleft,Aright,SH) = applyGateAndTrim(Aleft,Aright)
        A[row,col] = Aleft
        A[row,col+1] = Aright
        SH[row,col] = SH
        merge(row,col,UP,true)
        merge(row,col,DOWN,true)
        merge(row,col,LEFT,true)
        merge(row,col+1,UP,true)
        merge(row,col+1,RIGHT,true)
        merge(row,col+1,DOWN,true)

end

function applyGateAndUpdateDown(g, row, col)

        merge(row,col,UP,false)
        merge(row,col,DOWN,false)
        merge(row,col,LEFT,false)
        merge(row,col,RIGHT,false)
        merge(row+1,col,LEFT,false)
        merge(row+1,col,RIGHT,false)
        merge(row+1,col,DOWN,false)
        Aleft = A[row,col]
        Aright = A[row+1,col]
        (Aleft,Aright) = rotateTensors(Aleft,Aright)
        (Aleft,Aright,SH) = applyGateAndTrim(Aleft,Aright)
        (Aleft,Aright) = rotateTensorsBack(Aleft,Aright)
        A[row,col] = Aleft
        A[row+1,col] = Aright
        SH[row,col] = SH
        merge(row,col,UP,false)
        merge(row,col,DOWN,true)
        merge(row,col,LEFT,true)
        merge(row+1,col,LEFT,true)
        merge(row+1,col,RIGHT,true)
        merge(row+1,col,DOWN,true)

end

function rotateTensors(Ap,Bp)

    ap = size(Ap)
    bp = size(Bp)

    A2 = [Ap[a,b,c,d,s] for b = 1:bp[2], c = 1:bp[3], d = 1:bp[4], a = 1:bp[1], s = 1:pd]
    B2 = [Bp[a,b,c,d,s] for b = 1:ap[2], c = 1:ap[3], d = 1:ap[4], a = 1:ap[1], s = 1:pd]

    return(A2,B2)

end

function rotateTensorsBack(Ap,Bp)

    ap = size(Ap)
    bp = size(Bp)

    A2 = [Ap[a,b,c,d,s] for d = 1:bp[4], a = 1:bp[1], b = 1:bp[2], c = 1:bp[3], s = 1:pd]
    B2 = [Bp[a,b,c,d,s] for d = 1:ap[4], a = 1:ap[1], b = 1:ap[2], c = 1:ap[3], s = 1:pd]

    return(A2,B2)

end


function applyGateAndTrim(Aleft,Aright)

        @tensor begin
          ABg[a,e,f,s1p,b,c,d,s2p] := Aleft[a,x,e,f,s1]*Aright[b,c,d,x,s2]*g[s1,s2,s1p,s2p]
        end
        a = size(ABg)
        ABg = reshape(ABg,a[1]*a[2]*a[3]*pd,a[5]*a[6]*a[7]*pd)
        (U,d,V) = svd(ABg)
        newDim = min(D,length(d))
        U = U[:,1:newDim]
        V = V[:,1:newDim]
        newSH = diagm(d[1:newDim])
        A2p = reshape(U,a[1],a[2],a[3],pd,newDim)
        B2p = reshape(V',newDim,a[5],a[6],a[7],pd)
        A2p = [A2p[i,j,k,s,l] for i=1:a[1], l=1:newDim, j=1:a[2], k=1:a[3], s=1:pd]
        B2p = [B2p[i,j,k,l,s] for j=1:a[5], k=1:a[6], l=1:a[7], i=1:newDim, s=1:pd]
        return(A2p, B2p, newSH)
end



function merge(row, col, dir, inv)
    a = size(A[row,col])
    Arc = A[row,col]
    if dir == UP && row > 1
        SVrc = SV[row-1,col]
        if inv SVrc = diagm(inv.(diag(SVrc))) end
        @tensor begin
            temp[newA,b,c,d,s] := Arc[a,b,c,d,s] * SVrc[newA,a]
        end
        A[row,col] = temp
    elseif dir == DOWN && row < N
        SVrc = SV[row,col]
        if inv SVrc = diagm(inv.(diag(SVrc))) end
        @tensor begin
            temp[a,b,newC,d,s] := Arc[a,b,c,d,s] * SVrc[c,newC]
        end
        A[row,col] = temp
    elseif dir == RIGHT && col < N
        SHrc = SH[row,col]
        if inv SHrc = diagm(inv.(diag(SHrc))) end
        @tensor begin
            temp[a,newB,n,d,s] := Arc[a,b,c,d,s] * SHrc[b,newB]
        end
        A[row,col] = temp
    elseif dir == LEFT && col > 1
        SHrc = SH[row,col-1]
        if inv SHrc = diagm(inv.(diag(SHrc))) end
        @tensor begin
            temp[a,b,c,newD,s] := Arc[a,b,c,d,s] * SHrc[newD,d]
        end
        A[row,col] = temp
    end
end
