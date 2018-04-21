
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

  maxDim = maximum([size(newRow[k])[3] for k=1:N-1])
  #if (maxDim > Dp)
  if (row > 1 && row < N)
    RowEnv[row,:] = approxMPS2(newRow,Dp)
  else
    RowEnv[row,:] = newRow
  end

end
