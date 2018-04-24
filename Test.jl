function studyMPS(Big)

  New = [copy(Big[j]) for j = 1:N]
  halfN = Int64(ceil(N/2))


  for col = 1:N
      mid = mod(col+halfN,N)
        #initialize left and right
      for j = 1:mid-1
          li = mod(mid+j-1,N)+1
          ri = mod(mid-j,N)+1
          Newli = New[li]
          Newri = New[ri]
          Newliconj = conj.(Newli)
          Newriconj = conj.(Newri)
          @tensor begin
            NewLeft[x,y] := Newliconj[u,s,x]*Newli[w,s,y]*left[u,w]
            NewRight[u,w] := Newriconj[u,s,x]*Newri[w,s,y]*right[x,y]
          end
          left = NewLeft
          right = Newright
      end
      left = .5*(left + left')
      right = .5*(right+right')
      left = renormL2(left)
      right = renormL2(right)
      eigl = eigs(left)
      dl = eigl[1]
      vl = eigl[2]
      eigr = eigs(right)
      dr = eigr[1]
      vr = eigr[2]

      colp1 = mod(col,N) + 1
      colp2 = mod(col+1,N) + 1
      colm1 = mod(col-2,N) + 1

      R1 = reshape(New[colp1],size(New[colp1],1)*size(New[colp1],2),size(New[colp1],3))
      R2 = reshape(New[colp2],size(New[colp2],1),size(New[colp2],2)*size(New[colp2],3))
      R1 = R1*vr*diagm(sqrt.(dr))
      R2 = diagm(inv.(sqrt.(dr)))*vr'*R2
      New[colp2] = reshape(R2,size(R2,1),size(New[colp2],2),size(New[colp2],3))
      R1 = reshape(R1,size(New[colp1],1),size(New[colp1],2)*size(R1,2))

      L1 = reshape(New[colm1],size(New[colm1],1)*size(New[colm1],2),size(New[colm1],3))
      L2 = reshape(New[col],size(New[col],1),size(New[col],2)*size(New[col],3))
      L1 = L1*vl*diagm(sqrt.(dl))
      L2 = diagm(inv.(sqrt.(dl)))*vl'*L2
      New[colm1] = reshape(L1,size(New[colm1],1),size(New[colm1],2),size(L1,2))
      L2 = reshape(L2,size(L2,2)*size(New[col],2),size(New[col],3))

      both = L2*R1
      (U,d,V,trunc) = dosvdtrunc(both,Dp)
      dim = length(d)
      New[col] = reshape(U,size(New[col],1),size(New[col],2),dim)
      New[colp1] = reshape(diagm(d)*V,dim,size(New[colp1],2),size(New[colp1],3))

  end

  normBig = calcOverlapCycle(Big,Big)
  normNew = calcOverlapCycle(New,New)
  overlap = calcOverlapCycle(Big,New)
  @show((normBig+normNew-2*real(overlap))/normBig)

  return(New)
end

function getInnerProductOpen(T)

  T1conj = conj.(T[1])
  T1 = T[1]
  @tensor begin
    left[u,x,w,y] := T1conj[u,s,x]*T1[w,s,y]
  end
  for i = 2:N
    Ticonj = conj.(T[i])
    Ti = T[i]
    @tensor begin
      NewLeft[a,x,b,y] := Ticonj[u,s,x]*Ti[w,s,y]*left[a,u,b,w]
    end
    left = NewLeft
  end
  l = size(left)
  return(reshape(left,l[1]*l[2],l[3]*l[4]))

end

function getInnerProductClosed(T)

  left = eye(size(T[1],1))
  for i = 1:N
    Ticonj = conj.(T[i])
    Ti = T[i]
    @tensor begin
      NewLeft[x,y] := Ticonj[u,s,x]*Ti[w,s,y]*left[u,w]
    end
    left = NewLeft
  end
  return(left)

end


function testApproxMPS()

  D = 20
  pd = 6

  Big = [zeros(D,pd,D) for j = 1:N]
  Big[1] = zeros(1,pd,D)
  Big[N] = zeros(D,pd,1)

  for j = 1:N
    if !iseven(j)
      for d = 1:pd
        Big[j][1,d,d]=1
        Big[j+1][d,d,1]=1
      end
    end
  end

  approxMPS(Big,4)

end

function testDiag()

    for j = 1:N-1
        for k = 1:N
            sv = size(SV[j,k])[1]
            SV[j,k] = .01*eye(sv)
            SV[j,k][1,1] = 1
            sh = size(SH[k,j])[1]
            SH[k,j] = .01*eye(sh)
            SH[k,j][1,1] = 1
        end
    end
end

function test()

    New = [rand(Dp,4,Dp) for j = 1:N]
    New[1] = rand(1,4,Dp)
    New[N] = rand(Dp,4,1)

    testOverlap2(New,New)

end

function testOverlap(T,S,k)

  left = eye(1)
  for i = 1:k
    Siconj = conj.(S[i])
    Ti = T[i]
    @tensor begin
      NewLeft[x,y] := Siconj[u,s,x]*Ti[w,s,y]*left[u,w]
    end
    left = NewLeft
  end
  right = eye(1)
  for i = N:-1:k+1
    Siconj = conj.(S[i])
    Ti = T[i]
    @tensor begin
      NewRight[u,w] := Siconj[u,s,x]*Ti[w,s,y]*right[x,y]
    end
    right = NewRight
  end

  norm = trace(left*transpose(right))
  return(norm)

end

function testNorms()
  for j = 1:N
    for k = 1:N
      a = size(A[j,k])
      Avec = reshape(A[j,k],prod(a))
      norm = Avec'*Avec
      println("Norm of ($j,$k) = $norm")
    end
  end
end

function testOverlap2(T,S)

  M = [eye(2) for j=1:N]

  for i = 1:N
    SiVec = reshape(S[i],prod(size(S[i])))
    TiVec = reshape(T[i],prod(size(T[i])))
    if (length(SiVec) == length(TiVec))
      @show(SiVec'*TiVec)
    end
  end

  for i = 1:N
    Siconj = conj.(S[i])
    Ti = T[i]
    @tensor begin
      NewM[u,w,x,y] := Siconj[u,s,x]*Ti[w,s,y]
    end
    nm = size(NewM)
    M[i] = reshape(NewM,nm[1]*nm[2],nm[3]*nm[4])
  end



  for split = 0:N
      Left = eye(size(M[1])[1])
      Right = eye(size(M[N])[2])
      for j = 1:split
          Left = Left*M[j]
          tj = size(T[j])
          sj = size(S[j])
          if (tj[3] == sj[3])
            @show(trace(reshape(Left,tj[3],tj[3])))
          end
      end
      for j = N:-1:split+1
          Right = M[j]*Right
          tj = size(T[j])
          sj = size(S[j])
          if (tj[1] == sj[1])
            @show(trace(reshape(Right,tj[1],tj[1])))
          end
      end
      final = Left*Right
      @show(final)
  end

  final = M[1]
  for j = 2:N
      final = final*M[j]
  end
  @show(trace(final))

end
