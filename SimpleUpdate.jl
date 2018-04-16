
using TensorOperations
using LinearMaps

include("Contract.jl")
include("Test.jl")

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


pd = 2
N = 4
D = 3
Dp = 10
A = [zeros(1,1,1,1,pd) for j=1:N,  k = 1:N]
for j = 1:N
    for k = 1:N
        idx = (iseven(j+k)? 1: 2)
        A[j,k][1,1,1,1,idx] = 1
    end
end
SV = [eye(1) for j = 1:N-1, k = 1:N]
SH = [eye(1) for j = 1:N, k = 1:N-1]



RowEnv = [ones(1,1,1) for j=1:N, k=1:N]
SideEnv = [ones(1,1,1,1) for k=1:N]
endRow = [ones(1,1,1) for j=1:N]
endSide = ones(1,1,1,1)


#Global variables
sz = Float64[0.5 0; 0 -0.5]
sp = Float64[0 1; 0 0]
sm = sp'
Htwosite = reshape(JK(sz,sz) + 0.5 * JK(sp,sm) + 0.5 * JK(sm,sp),2,2,2,2)
lambda = 3.0
sigZ = Float64[1 0; 0 -1]
sigX = Float64[0 1; 1 0]
#Htwosite = reshape(JK(sigZ,sigZ) + lambda*0.25*JK(eye(2),sigX) + lambda*0.25*JK(sigX,eye(2)),2,2,2,2)
# order for Htwosite is s1, s2, s1p, s2p



function mainLoop()
    numIter = 100
    tau = .2
    for iter = 1:numIter
        taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
        println("\n iteration = $iter")
        for j = 1:N
            for k = 1:N-1
                applyGateAndUpdateRight(taugate, j, k) #true = horiz
                applyGateAndUpdateDown(taugate, k, j) #false = vert
            end
        end
    end
    @show("Calculating Energy")
    calcEnergy()
end

function applyGateAndUpdateRight(g, row, col)

#println("\n Updating Right: Row = $row,  Col = $col")
        Aleft = A[row,col]
        Aright = A[row,col+1]
        Aleft = merge(Aleft,row,col,UP,false)
        Aleft = merge(Aleft,row,col,DOWN,false)
        Aleft = merge(Aleft,row,col,LEFT,false)
        Aleft = merge(Aleft,row,col,RIGHT,false)
        Aright = merge(Aright,row,col+1,UP,false)
        Aright = merge(Aright,row,col+1,RIGHT,false)
        Aright = merge(Aright,row,col+1,DOWN,false)
        (Aleft,Aright,SHnew) = applyGateAndTrim(Aleft,Aright,g)
        Aleft = merge(Aleft,row,col,UP,true)
        Aleft = merge(Aleft,row,col,DOWN,true)
        Aleft = merge(Aleft,row,col,LEFT,true)
        Aright = merge(Aright,row,col+1,UP,true)
        Aright = merge(Aright,row,col+1,RIGHT,true)
        Aright = merge(Aright,row,col+1,DOWN,true)
        A[row,col] = Aleft
        A[row,col+1] = Aright
        SH[row,col] = SHnew

end

function applyGateAndUpdateDown(g, row, col)

  #Sprintln("\n Updating Down: Row = $row,  Col = $col")
        Aup = A[row,col]
        Adown = A[row+1,col]
        Aup = merge(Aup,row,col,UP,false)
        Aup = merge(Aup,row,col,DOWN,false)
        Aup = merge(Aup,row,col,LEFT,false)
        Aup = merge(Aup,row,col,RIGHT,false)
        Adown = merge(Adown,row+1,col,LEFT,false)
        Adown = merge(Adown,row+1,col,RIGHT,false)
        Adown = merge(Adown,row+1,col,DOWN,false)
        (Aup,Adown) = rotateTensors(Aup,Adown)
        (Aup,Adown,SVnew) = applyGateAndTrim(Aup,Adown,g)
        (Aup,Adown) = rotateTensorsBack(Aup,Adown)
        Aup = merge(Aup,row,col,UP,true)
        Aup = merge(Aup,row,col,RIGHT,true)
        Aup = merge(Aup,row,col,LEFT,true)
        Adown = merge(Adown,row+1,col,LEFT,true)
        Adown = merge(Adown,row+1,col,RIGHT,true)
        Adown = merge(Adown,row+1,col,DOWN,true)
        A[row,col] = Aup
        A[row+1,col] = Adown
        SV[row,col] = SVnew

end

function rotateTensors(Ap,Bp)

    ap = size(Ap)
    bp = size(Bp)

    A2 = [Ap[a,b,c,d,s] for b = 1:ap[2], c = 1:ap[3], d = 1:ap[4], a = 1:ap[1], s = 1:pd]
    B2 = [Bp[a,b,c,d,s] for b = 1:bp[2], c = 1:bp[3], d = 1:bp[4], a = 1:bp[1], s = 1:pd]

    return(A2,B2)

end

function rotateTensorsBack(Ap,Bp)

    ap = size(Ap)
    bp = size(Bp)

    A2 = [Ap[a,b,c,d,s] for d = 1:ap[4], a = 1:ap[1], b = 1:ap[2], c = 1:ap[3], s = 1:pd]
    B2 = [Bp[a,b,c,d,s] for d = 1:bp[4], a = 1:bp[1], b = 1:bp[2], c = 1:bp[3], s = 1:pd]

    return(A2,B2)

end


function applyGateAndTrim(Aleft,Aright,g)

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
        A2p = renormL2(A2p)
        B2p = renormL2(B2p)
        newSH = renormL2(newSH)
        A2p = [A2p[i,j,k,s,l] for i=1:a[1], l=1:newDim, j=1:a[2], k=1:a[3], s=1:pd]
        B2p = [B2p[i,j,k,l,s] for j=1:a[5], k=1:a[6], l=1:a[7], i=1:newDim, s=1:pd]
        return(A2p, B2p, newSH)
end



function merge(Arc,row, col, dir, doInv)
    a = size(Arc)
    temp = ones(1,1,1,1,1)
    if dir == UP && row > 1
        SVrc = SV[row-1,col]
        if (doInv) SVrc = diagm(inv.(diag(SVrc))) end
        @tensor begin
            temp[newA,b,c,d,s] := Arc[a,b,c,d,s] * SVrc[newA,a]
        end
        return(temp)
    elseif dir == DOWN && row < N
        SVrc = SV[row,col]
        if (doInv) SVrc = diagm(inv.(diag(SVrc))) end
        @tensor begin
            temp[a,b,newC,d,s] := Arc[a,b,c,d,s] * SVrc[c,newC]
        end
        return(temp)
    elseif dir == RIGHT && col < N
        SHrc = SH[row,col]
        if (doInv) SHrc = diagm(inv.(diag(SHrc))) end
        @tensor begin
            temp[a,newB,c,d,s] := Arc[a,b,c,d,s] * SHrc[b,newB]
        end
        return(temp)
    elseif dir == LEFT && col > 1
        SHrc = SH[row,col-1]
        if (doInv) SHrc = diagm(inv.(diag(SHrc))) end
        @tensor begin
            temp[a,b,c,newD,s] := Arc[a,b,c,d,s] * SHrc[newD,d]
        end
        return(temp)
    end
    return(Arc)
end

function renormL2(T)
  t = size(T)
  Tvec = reshape(T,prod(t))
  norm = abs(Tvec'*Tvec)
  T = T/sqrt(norm)
  return(T)
end
