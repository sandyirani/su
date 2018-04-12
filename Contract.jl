function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
    (a1,a2) = size(a)
    (b1,b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
end


function calcEnergy()
  energy = 0
  initRowEnv()
  for row = N:-1:1
    for col=N:-1:3
      updateSideEnvToLeft(row, col)
    end
    for col = 1:N-1
      energy += contractTwoSite(row,col,true)
      norm = contractTwoSite(row,col,false)
      @show(norm)
    end
  end
  return(energy)
end

function contractTwoSite(row,col,addEnergy)
  Tlp = conj.(A[row,col])
  Trp = conj.(A[row,col+1])
  if addEnergy
    (Tlpg,Trpg) = applyGate(Tlp,Trp,Htwosite)
  else
    (Tlpg,Trpg) = (Tlp,Trp)
  end

  leftSide = (col == 1? endSide: SideEnv[col-1])
  newSide = updateSideEnvToRight(leftSide, row, col, A[row,col], Tlpg)
  newSide = updateSideEnvToRight(newSide, row, col+1, A[row,col+1], Trpg)
  rightSide = (col == N-1? endSide: SideEnv[col+2])
  rightSideVec = reshape(rightSide,prod(size(rightSideVec)))
  newSideVec = reshape(rightSide,prod(size(newSideVec)))
  energy = newSideVec'*rightSideVec
  return(energy)
end

function applyGate(Tl,Tr,g)
  @tensor begin
    Tg[a,e,f,s1p,b,c,d,s2p] := Tl[a,x,e,f,s1]*Tr[b,c,d,x,s2]*g[s1,s2,s1p,s2p]
  end
  tg = size(Tg)
  Tg = reshape(Tg,prod(tg[1:4]),prod(tg[5:8]))
  (U,d,V) = svd(Tg)
  U = U * diagm(d)
  newDim = length(d)
  Tl1 = reshape(U,tg[1],tg[2],tg[3],tg[4],newDim)
  Tr1 = reshape(V',newDim,tg[5],tg[6],tg[7],tg[8])
  Tl2 = [Tl1[i,j,k,s,l] for i=1:tg[1], l=1:newDim, j=1:tg[2], k=1:tg[3], s=1:tg[4]]
  Tr2 = [Tr1[i,j,k,l,s] for j=1:tg[5], k=1:tg[6], l=1:tg[7], i=1:newDim, s=1:tg[8]]
  return(Tl2, Tr2)
end

function initRowEnv()
  for j = 1:N-1
    updateRowEnv(j,true)
  end
end

function updateSideEnvToRight(leftSide, row, col, T, Tp)

  newSide = ones(1,1,1,1)

  dimN = (row == 1? 1:D)
  dimS = (row == N? 1:D)
  upEnv = (row == 1? endRow[col]: RowEnv[row-1,col])
  downEnv = (row == N? endRow[col]: RowEnv[row+1,col])
  ue = size(upEnv)
  de = size(downEnv)
  ls = size(leftSide)

  temp = reshape(rightSide,ls[1]*ls[2]*ls[3],ls[4])*reshape(downEnv,de[1],de[2]*de[3])
  temp = reshape(temp,ls[1],ls[2],ls[3],de[2],de[3])
  temp = reshape(temp,ls[1],ls[2]*ls[3]*de[2]*de[3])
  temp2 = transpose(temp)*reshape(upEnv,ue[1],ue[2]*ue[3])
  temp2 = reshape(temp2,ls[2],ls[3],dimS,dimS,de[3],dimN,dimN,ue[3])


  @tensor begin
    temp3[x,bp,b,y] := temp2[dp,d,cp,c,y,ap,a,x]*T[a,b,c,d,s]*Tp[ap,bp,cp,dp,s]
  end
  return(temp3)

end

function updateSideEnvToLeft(row, col)

  newSide = ones(1,1,1,1)
  lastSide = (col == N? endSide: SideEnv[col-1])

  dimN = (row == 1? 1:D)
  dimS = (row == N? 1:D)
  upEnv = (row == 1? endRow[col]: RowEnv[row-1,col])
  downEnv = (row == N? endRow[col]: RowEnv[row+1,col])
  ue = size(upEnv)
  de = size(downEnv)
  ls = size(lastSide)

  temp = transpose(reshape(downEnv,de[1]*de[2],de[3])*transpose(reshape(lastSide,ls[1]*ls[2]*ls[3],ls[4])))
  temp = reshape(temp,ls[1],ls[2],ls[3],de[1],de[2])
  temp2 = reshape(upEnv,ue[1]*ue[2],ue[3])*reshape(temp,ls[1],ls[2]*ls[3]*de[1]*de[2])
  temp2 = reshape(temp2,ue[1],dimN,dimN,ls[2],ls[3],de[1],dimS,dimS)

  T = A[row,col]
  Tp = conj.(A[row,col])
  @tensor begin
    temp3[x,dp,d,y] := temp2[x,ap,a,bp,b,y,cp,c]*T[a,b,c,d,s]*Tp[ap,bp,cp,dp,s]
  end
  return(temp3)

end

function updateRowEnv(row, topDown)

  newRow = [ones(1,1,1) for k = 1:N]
  if (topDown)
    lastRow = (row == 1? endRow: RowEnv[row-1,:])
  else
    lastRow = (row == N? endRow: RowEnv[row+1,:])
  end
  dim = (row ==1 || row == N? 1:D)

  for k = 1:N
    RE = lastRow[k]
    re = size(RE)
    RE = reshape(RE,re[1],dim,dim,re[3])
    T = A[row,k]
    Tconj = conj.(A[row,k])
    if (topDown)
      @tensor begin
        NewRE[a,fp,f,ep,e,c,dp,d] := RE[a,bp,b,c]*T[b,d,e,f,s]*T[bp,dp,ep,fp,s]
      end
    else
      @tensor begin
        NewRE[fp,f,a,dp,d,ep,e,c] := RE[a,bp,b,c]*T[d,e,b,f,s]*T[dp,ep,bp,fp,s]
      end
    end
    nre = size(NewRE)
    NewRE = reshape(NewRE,nre[1]*nre[2]*nre[3],nre[4]*nre[5],nre[6]*nre[7]*nre[8])
    newRow[k] = newRE
  end

  if (row > 1 && row < N)
    RowEnv[row,:] = approx(newRow,Dp)
  else
    RowEnv[row,:] = newRow
  end

end


function approxMPS(Big,Dp)

  pd = [size(Big[j])[2] for j=1:N] #particle dimensions

  New = [rand(Dp,pd[j],Dp) for j = 1:N]
  New[1] = rand(1,pd[1],Dp)
  New[N] = rand(Dp,pd[N],1)

  BN = [eye(1) for j = 1:N]
  NN = [eye(1) for j = 1:N]

  for i = 1:N-1 #initialize BN and NN arrays
    BNim1 = (i>1? BN[i-1]: eye(1))
    NNim1 = (i>1? NN[i-1]: eye(1))
    Biconj = conj.(Big[i])
    Ni = New[i]
    Niconj = conj.(Ni)
    @tensor begin
      BNi[x,y] := Biconj[u,s,x]*Ni[w,s,y]*BNim1[u,w]
      NNi[x,y] := Niconj[u,s,x]*Ni[w,s,y]*NNim1[u,w]
    end
    BN[i] = BNi
    NN[i] = NNi
  end


  NormB = calcNorm(Big)
  @show(NormB)


  for iter = 1:20 #main Loop

    for ii=-N:N		# if negative, going right to left

      ii == 0 && continue
      i = abs(ii)
      toright = ii > 0

      LeftNN = (i>1? NN[i-1]: eye(1))
      RightNN = (i<N? NN[i+1]: eye(1))
      LeftBN = (i>1? BN[i-1]: eye(1))
      RightBN = (i<N? BN[i+1]: eye(1))

      l = size(LeftBN)
      r = size(RightBN)

      R = JK(JK(LeftNN, eye(pd[i])),RightNN)
      R = 0.5*(R+R')
      S = reshape(transpose(LeftBN)*reshape(Big[i],l[1],pd[i]*r[1]),l[2]*pd[i],r[1])
      S = reshape(S*RightBN,l[2]*pd[i]*r[2])

      newiVec = \(R,S)
      #i == N-1 && @show(sum(abs.(R*newiVec-S))/sum(abs.(S)))
      dist = NormB + newiVec'*R*newiVec - 2*real(S'*newiVec)
      i == N-1 && @show(dist/NormB)

      New[i] = reshape(newiVec,l[2],pd[i],r[2])
      Biconj = conj.(Big[i])
      Ni = New[i]
      Niconj = conj.(Ni)

      if toright
        @tensor begin
          BNi[x,y] := Biconj[u,s,x]*Ni[w,s,y]*LeftBN[u,w]
          NNi[x,y] := Niconj[u,s,x]*Ni[w,s,y]*LeftNN[u,w]
        end
      else
        @tensor begin
          BNi[u,w] := Biconj[u,s,x]*Ni[w,s,y]*RightBN[x,y]
          NNi[u,w] := Niconj[u,s,x]*Ni[w,s,y]*RightNN[x,y]
        end
      end
      BN[i] = BNi
      NN[i] = NNi

    end
  end

end

function calcNorm(A)

  left = eye(1)
  for i = 1:N
    Aiconj = conj.(A[i])
    Ai = A[i]
    @tensor begin
      NewLeft[x,y] := Aiconj[u,s,x]*Ai[w,s,y]*left[u,w]
    end
    left = NewLeft
  end
  norm = trace(left)
  return(norm)

end
