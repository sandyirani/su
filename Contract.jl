function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
    (a1,a2) = size(a)
    (b1,b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
end


RowEnv = [ones(1,1,1) for i=1:N+2, for j=1:N]

function updateRowEnv(row)

  if (row < 1 || row > N) return;

  newRow = [ones(1,1,1) for k = 1:N]
  for k = 1:N
    RE = RowEnv[row,k]
    T = A[row,k]
    @tensor begin
      NewRE[a,f,e,c,d] := RE[a,b,c]*T[b,d,e,f]
    end
    nre = size(NewRE)
    NewRE = reshape(NewRE,nre[1]*nre[2],nre[3],nre[4]*nre[5])
  end
  if k > 2
    RE[row+1,k] = approx(NewRE,D)
  else
    RE[row+1,k] = NewRE  
  end

end


function approxMPS(Big,D)

  pd = [size(Big[j])[2] for j=1:N] #particle dimensions

  New = [rand(D,pd[j],D) for j = 1:N]
  New[1] = rand(1,pd[1],D)
  New[N] = rand(D,pd[N],1)

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
