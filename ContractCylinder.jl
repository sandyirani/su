
function mergeRows()
  AM = mergeA()
  for j = 1:N-1
    updateRowEnv(AM,j,true)
  end
end

function mergeA()
    AM = [zeros(1,1,1,1,pd) for j=1:N,  k = 1:N]
    for row = 1:N
        for col = 1:N
            temp = A[row,col]
            if (row < N)
                temp = merge(temp,row,col,DOWN,false)
            end
            #Cyl
            temp = merge(temp,row,col,RIGHT,false)
            AM[row,col] = temp
        end
    end
    return(AM)
end



function updateRowEnv(AM,row, topDown)

  newRow = [ones(1,1,1) for k = 1:N]
  if (topDown)
    lastRow = (row == 1? endRow: RowEnv[row-1,:])
  else
    lastRow = (row == N? endRow: RowEnv[row+1,:])
  end
  dim = (row ==1 || row == N? 1:D)
  newRE = ones(1,1,1)

  for k = 1:N
    RE = lastRow[k]
    re = size(RE)
    RE = reshape(RE,re[1],dim,dim,re[3])
    T = AM[row,k]
    Tconj = conj.(AM[row,k])
    if (topDown)
      @tensor begin
        newRE[a,fp,f,ep,e,c,dp,d] := RE[a,bp,b,c]*T[b,d,e,f,s]*T[bp,dp,ep,fp,s]
      end
    else
      @tensor begin
        newRE[fp,f,a,dp,d,ep,e,c] := RE[a,bp,b,c]*T[d,e,b,f,s]*T[dp,ep,bp,fp,s]
      end
    end
    nre = size(newRE)
    newRE = reshape(newRE,nre[1]*nre[2]*nre[3],nre[4]*nre[5],nre[6]*nre[7]*nre[8])
    newRow[k] = newRE
  end


  if (D^row > Dp && row < N)
    RowEnv[row,:] = approxMPS2(newRow,Dp)
  else
    RowEnv[row,:] = newRow
  end

end

function approxMPS2(Big,Dp)

  New = [ones(1,1,1) for j = 1:N]

  for j = N-1:-1:2
    left = Big[j]
    right = (j < N-1? New[j+1]: Big[j+1])
    l = size(left)
    r = size(right)
    left = reshape(left,l[1]*l[2],l[3])
    right = reshape(right,r[1],r[2]*r[3])
    both = left*right
    (U,d,V,trunc) = dosvdtrunc(both,l[3])
    dim = length(d)
    New[j] = reshape(U*diagm(d),l[1],l[2],dim)
    New[j+1] = reshape(V,dim,r[2],r[3])
  end

  for j = 1:N-1
    left = (j > 1? New[j]: Big[j])
    right = Big[j+1]
    l = size(left)
    r = size(right)
    left = reshape(left,l[1]*l[2],l[3])
    right = reshape(right,r[1],r[2]*r[3])
    both = left*right
    (U,d,V,trunc) = dosvdtrunc(both,Dp)
    dim = length(d)
    New[j] = reshape(U,l[1],l[2],dim)
    New[j+1] = reshape(diagm(d)*V,dim,r[2],r[3])
  end

  normBig = calcNorm(Big)
  normNew = calcNorm(New)
  overlap = calcOverlap(Big,New)
  @show((normBig+normNew-2*real(overlap))/normBig)

  return(New)

end
