
function mergeRows()
  AM = mergeA()
  for j = 1:N-1
    println("\n Merging row $j")
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

  if (D^(2*row) > Dp && row < N)
    RowEnv[row,:] = approxMPS2(newRow,Dp)
  else
    RowEnv[row,:] = newRow
  end


end

function approxMPS2(Big,Dp)

  New = [copy(Big[j]) for j = 1:N]
  halfN = Int64(ceil(N/2))

  for col = 1:N
      colp1 = mod(col,N) + 1
      mid = mod(col+halfN,N)
      for j = 1:mid-1
          li = mod(mid+j-1,N)+1
          ri = mod(li,N)+1
          (New[li],New[ri])  = moveHoriz(New[li],New[ri],size(New[li])[3],false)
          li = mod(mid-j-1,N)+1
          ri = mod(li,N)+1
          @show(size(New[li]),size(New[ri]))
          (New[li],New[ri])  = moveHoriz(New[li],New[ri],size(New[li])[3],true)
      end
      (New[col],New[colp1])  = moveHoriz(New[col],New[colp1],Dp,false)
  end

  normBig = calcOverlapCycle(Big,Big)
  normNew = calcOverlapCycle(New,New)
  overlap = calcOverlapCycle(Big,New)
  @show((normBig+normNew-2*real(overlap))/normBig)

  return(New)

end

function moveHoriz(left,right,m,toLeft)

    l = size(left)
    r = size(right)
    left = reshape(left,l[1]*l[2],l[3])
    right = reshape(right,r[1],r[2]*r[3])
    both = left*right
    (U,d,V,trunc) = dosvdtrunc(both,m)
    dim = length(d)
    if (toLeft)
        left = reshape(U*diagm(d),l[1],l[2],dim)
        right = reshape(V,dim,r[2],r[3])
    else
        left = reshape(U,l[1],l[2],dim)
        right = reshape(diagm(d)*V,dim,r[2],r[3])
    end
    return(left,right)

end
