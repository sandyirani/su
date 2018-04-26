function cleanEigs(d,v)
    count = 0
    for j = 1:length(d)
        count = (d[j]>0? count+1: count)
    end
    dNew = zeros(count)
    vNew = zeros(size(v,1),count)
    curr = 1
    for j = 1:length(d)
        if (d[j]>0)
            dNew[curr] = d[j]
            vNew[:,curr] = v[:,j]
            curr += 1
        end
    end
    return(dNew,vNew)
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

function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
    (a1,a2) = size(a)
    (b1,b2) = size(b)
    reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
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

function renormL2(T)
  t = size(T)
  Tvec = reshape(T,prod(t))
  norm = abs(Tvec'*Tvec)
  T = T/sqrt(norm)
  return(T)
end

function calcNorm(T)

  left = eye(size(T[1],1))
  for i = 1:N
    Ticonj = conj.(T[i])
    Ti = T[i]
    @tensor begin
      NewLeft[x,y] := Ticonj[u,s,x]*Ti[w,s,y]*left[u,w]
    end
    left = NewLeft
  end
  norm = trace(left)
  return(norm)

end

function calcOverlap(T,S)

  left = eye(1)
  for i = 1:N
    Siconj = conj.(S[i])
    Ti = T[i]
    @tensor begin
      NewLeft[x,y] := Siconj[u,s,x]*Ti[w,s,y]*left[u,w]
    end
    left = NewLeft
  end
  norm = trace(left)
  return(norm)

end

function calcOverlapCycle(T,S)

  left = eye(size(T[1])[1]*size(S[1])[1])
  for i = 1:N
    Siconj = conj.(S[i])
    Ti = T[i]
    @tensor begin
      NewM[u,w,x,y] := Siconj[u,s,x]*Ti[w,s,y]
    end
    nm = size(NewM)
    left = left*reshape(NewM,nm[1]*nm[2],nm[3]*nm[4])
  end
  norm = trace(left)
  return(norm)

end

function dosvdtrunc(AA,m)		# AA a matrix;  keep at most m states

    (u,d,v) = svd(AA)

    prob = dot(d,d)		# total probability
    mm = min(m,length(d))	# number of states to keep
    d = d[1:mm]			# middle matrix in vector form
    trunc = prob - dot(d,d)
    U = u[:,1:mm]
    V = v[:,1:mm]'
    (U,d,V,trunc)		# AA == U * diagm(d) * V	with error trunc
end

function rotateTensor(T)

    t = size(T)

    T2 = [T[a,b,c,d,s] for b = 1:t[2], c = 1:t[3], d = 1:t[4], a = 1:t[1], s = 1:pd]

    return(T2)

end

function rotateGrid(AM)
  AMnew = [zeros(1,1,1,1,pd) for j=1:N,  k = 1:N]
  for row = 1:N
    for col = 1:N
      AMnew[row,col] = rotateTensor(AM[col,N-row+1])
    end
  end
  return(AMnew)
end
