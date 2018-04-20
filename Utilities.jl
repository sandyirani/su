
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

  left = eye(1)
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
