function studyMPS(Big)
  @show(calcOverlapCycle(Big))
  Line = [ones(1,1,1) for j = 1:N+2]
  left = zeros(1,size(Big[1],1),size(Big[1],1))
  right = zeros(size(Big[N],3),size(Big[N],3),1)
  left[1,:,:] = eye(size(Big[1],1))
  right[:,:,1] = eye(size(Big[1],1))
  Line[1] = left
  Line[N+2] = right
  for k = 1:N
    for j = 1:N
      Line[j+1] = Big[mod(j+k-2,N)+1]
      @show(calcNorm(Line))
    end
  end
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
