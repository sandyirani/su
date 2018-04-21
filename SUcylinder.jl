
using TensorOperations
using LinearMaps

include("Utilities.jl")
include("ContractCylinder.jl")
include("Test.jl")

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


pd = 2
N = 10
D = 2
Dp = 10
A = [zeros(1,1,1,1,pd) for j=1:N,  k = 1:N]
for j = 1:N
    for k = 1:N
        idx = (iseven(j+k)? 1: 2)
        A[j,k][1,1,1,1,idx] = 1
    end
end
SV = [eye(1) for j = 1:N-1, k = 1:N]
SH = [eye(1) for j = 1:N, k = 1:N]


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
  #numIters = [1000,2000,8000]
  numIters = [100,200,800]
  taus = [.1,.01,.001]
  for stage = 1:3
    tau = taus[stage]
    numIter = numIters[stage]
    for iter = 1:numIter
      taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
      #println("\n iteration = $iter")
      for j = 1:N
        for k = 1:N
          applyGateAndUpdateRight(taugate, j, k) #true = horiz
          if (k < N)
              applyGateAndUpdateDown(taugate, k, j) #false = vert
          end
        end
      end
    end
    println("\n End of stage $stage")
  end
  println("Merging Rows")
  mergeRows()
end

function applyGateAndUpdateRight(g, row, col)

        #println("\n Updating Right: Row = $row,  Col = $col")
        Aleft = A[row,col]
        colp1 = (col < N? col+1: 1)
        Aright = A[row,colp1]

        Aleft = merge(Aleft,row,col,UP,false)
        Aleft = merge(Aleft,row,col,DOWN,false)
        Aleft = merge(Aleft,row,col,LEFT,false)
        Aleft = merge(Aleft,row,col,RIGHT,false)
        Aright = merge(Aright,row,colp1,UP,false)
        Aright = merge(Aright,row,colp1,RIGHT,false)
        Aright = merge(Aright,row,colp1,DOWN,false)
        (Aleft,Aright,SHnew) = applyGateAndTrim(Aleft,Aright,g)
        Aleft = merge(Aleft,row,col,UP,true)
        Aleft = merge(Aleft,row,col,DOWN,true)
        Aleft = merge(Aleft,row,col,LEFT,true)
        Aright = merge(Aright,row,colp1,UP,true)
        Aright = merge(Aright,row,colp1,RIGHT,true)
        Aright = merge(Aright,row,colp1,DOWN,true)
        A[row,col] = Aleft
        A[row,colp1] = Aright
        SH[row,col] = SHnew

end

function applyGateAndUpdateDown(g, row, col)

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




function merge(Arc,row, col, dir, doInv)
    #println("\n Merge: Row = $row,  Col = $col  Dir = $dir")
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
    elseif dir == RIGHT
        SHrc = SH[row,col]
        if (doInv) SHrc = diagm(inv.(diag(SHrc))) end
        @tensor begin
            temp[a,newB,c,d,s] := Arc[a,b,c,d,s] * SHrc[b,newB]
        end
        return(temp)
    elseif dir == LEFT
        colm1 = (col > 1? col-1: N)
        SHrc = SH[row,colm1]
        if (doInv) SHrc = diagm(inv.(diag(SHrc))) end
        @tensor begin
            temp[a,b,c,newD,s] := Arc[a,b,c,d,s] * SHrc[newD,d]
        end
        return(temp)
    end
    return(Arc)
end
