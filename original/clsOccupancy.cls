VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "clsOccupancy"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
' ===============================================================================================
' ## Four-state Domestic Occupancy Model v1.1
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

Private intDwellingIndex As Integer

Private intResidents As Integer

' A flag to determine if a weekend run is required
Private blnWeekend As Boolean

' An array to store the combined states for a single day
Private aCombinedState(143, 0) As String

Private aActiveOccupancy(143, 0) As Integer

Private aOccupancyThermalGains(143, 0) As Double


' ===============================================================================================
' Class Properties

Public Property Get GetActiveOccupancy() As Integer()
    GetActiveOccupancy = aActiveOccupancy()
End Property

Public Property Get ThermalGains() As Double()
    ThermalGains = aOccupancyThermalGains()
End Property

' // Get occupancy casual thermal gain for a specific timestep
Public Property Get GetPhi_cOccupancy(tenMinute As Integer)
    GetPhi_cOccupancy = aOccupancyThermalGains(tenMinute, 0)
End Property
Public Property Get GetMeanActiveOccupancy() As Double
    GetMeanActiveOccupancy = WorksheetFunction.Average(aActiveOccupancy)
End Property

Public Property Get GetPrActivelyOccupied() As Double
    Dim aTest(143, 0) As Integer
    Dim intRow As Integer
    
    For intRow = 0 To 143
        aTest(intRow, 0) = IIf(aActiveOccupancy(intRow, 0) >= 1, 1, 0)
    Next intRow
    GetPrActivelyOccupied = WorksheetFunction.Average(aTest)
End Property




' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseOccupancy
'
' ===============================================================================================
Public Sub InitialiseOccupancy(index As Integer)
    
    Dim intOffset As Integer
    intOffset = 4
    ' // Declare variables
    
    Dim strCell As String ' A variable used to store the value in a cell
        
    ' // Get the input variables
    
    intDwellingIndex = index
    
    ' // Get the number of residents
    
    intResidents = Application.index(wsDwellings.Range("rNumberResidentsIndex"), intDwellingIndex + intOffset).Value
    
    ' // Perform a range check on the input resident count
        
    If (intResidents < 1 Or intResidents > 6) Then
        wsMain.Range("J18") = "Error - simulation stopped"
        MsgBox "Please enter the number of residents from 1 to 6.", vbExclamation
        End
    End If
    
    
    ' // Perform a range check on the weekday/weekend flag
    strCell = wsMain.Range("rDayType").Value
    If Not (strCell = "wd" Or strCell = "we") Then
        wsMain.Range("J18") = "Error - simulation stopped"
        MsgBox "Please specify either a weekday (wd) or a weekend (we).", vbExclamation
        End
    End If
    
    ' // Get the weekday or weekend flag
    blnWeekend = wsMain.Range("rDayType").Value = "we"
       

End Sub




' ===============================================================================================
' ## RunFourStateOccupancySimulation
'
' ===============================================================================================
Public Sub RunFourStateOccupancySimulation()
 
    ' // Declare variables
    
    ' An array to store a transition probability matrix
    Dim aTPM()
    
    ' An array to store a row of transition probabilities
    Dim aTPR()
    
    ' A number to store a row reference
    Dim intRow As Integer
    
    ' The total number of states
    Dim intPossibleStates As Integer
    
    ' A number to store a row reference
    Dim intLastRow As Integer
    
    ' A number to store a column reference
    Dim intLastColumn As Integer
    
    ' A loop count variable
    Dim intTimeStep As Integer
    
    ' A number to store a column reference
    Dim intCol As Integer
    
    ' A variable used to store the value in a cell
    Dim strCell As String
    
    ' A string to store a column reference
    Dim strCol As String
    
    ' A string to hold the combined state
    Dim strCombinedState As String
    
    ' A flag to determine if the transitions should be amended to
    Dim bln24HourOccupancyFix As Boolean
    
    ' // More accurately account for the proportion of dwellings that have 24 hour occupancy in the UK
    ' // time-use survey
    ' A flag to determine if a dwelling should have 24 hour occupancy
    Dim bln24HourOccupancyDwelling As Boolean
    
    ' A variable to store a random number
    Dim dblRand As Double
    
    ' A variable to store a cumulative probability
    Dim dblCumulativeP As Double
    
    ' A variable to store the uplift factor required to force dwellings into having 24 hour occupancy
    Dim dblUplift As Double
    
    ' A variable to store a sum
    Dim dblSum As Double
    
    ' A worksheet object to reference the relevant TPM sheet
    Dim wsTPM As Worksheet
    
    ' // Determine the number of possible occupancy states
    intPossibleStates = (intResidents + 1) * (intResidents + 1)
    
    ' // Allocate memory for an array to store a row of transition probabilities
    ReDim aTPR(1 To 1, 1 To intPossibleStates)
            
    ' // Determine the last row and column for the transition probability matrix
    intLastRow = 144 * intPossibleStates + 10
    intLastColumn = intPossibleStates + 2
    
    ' // Get a string for whether it's a weekend or weekday
    strCell = wsMain.Range("rDayType").Value

    ' // Get the reference to the required worksheet
    Set wsTPM = ThisWorkbook.Sheets("tpm" + CStr(intResidents) + "_" + CStr(strCell))
    
    ' // Get the transition probability matrix
    aTPM = wsTPM.Range("A10:" + CStr(Cells(intLastRow, intLastColumn).Address))

    ' // Determine the letter of the column from the starting states worksheet that corresponds to
    ' // the specified number of residents
    strCol = Left(wsStartingStates.Cells(2, 1 + intResidents).Address(True, False, xlA1), 1)
    
    ' // Initialise the random number generator
    Randomize
    
    ' // Determine the start state at time 00:00 by checking
    ' // a new random number against the distribution
    
    ' // Pick a random number to determine the start state
    dblRand = Rnd()
    
    ' // Reset the cumulative probability
    dblCumulativeP = 0
    
    For intRow = 0 To 48
        ' // Add the probability for this occupancy state
        dblCumulativeP = dblCumulativeP + wsStartingStates.Range(strCol + CStr(intRow + 7 + IIf(blnWeekend, 54, 0))).Value
        If dblRand < dblCumulativeP Then
            ' // This determines the starting occupancy state
            strCombinedState = wsStartingStates.Range("A" + CStr(intRow + 7 _
                + IIf(blnWeekend, 54, 0))).Value
            
            ' // Assign occupancy state to the first time step for this sequence
            aCombinedState(0, 0) = strCombinedState
            aActiveOccupancy(0, 0) = WorksheetFunction.Min(CInt(Left(strCombinedState, 1)), CInt(Right(strCombinedState, 1)))
            Exit For
            
        End If
        
    Next intRow
    
    ' // Get the uplift factor to force the right proportion of dwellings into having
    ' // 24 hour occupancy
    dblUplift = ws24hrOccupancy. _
        Cells(intResidents + 3, IIf(blnWeekend = False, 6, 7)).Value
    
    ' // Set the flag to determine if this is a dwelling that should have 24 hour occupancy
    bln24HourOccupancyDwelling = (Rnd() < dblUplift)
        
    For intTimeStep = 1 To 143
        ' // Get the appropriate transition probabilities, and amend if necessary
    
        ' // Determine the row in the transition probability for this time step and occupancy state
        intRow = 2 + (intTimeStep - 1) * intPossibleStates _
            + (intResidents + 1) _
            * IIf(Left(strCombinedState, 1) = "0", 0, CInt(Left(strCombinedState, 1))) _
            + CInt(Right(strCombinedState, 1))
        
        ' // Get the row of transition probabilities
        For intCol = 1 To intPossibleStates
            aTPR(1, intCol) = aTPM(intRow, intCol + 2)
        Next intCol
        
        ' // If this is a dwelling that requires 24 hour occupancy, then change the transition
        ' // probabilities such that transitions to unoccupied states cannot occur
        If bln24HourOccupancyDwelling = True Then
            ' // Sum the probability of transitioning to an unoccupied state
            dblSum = 0
            For intCol = 1 To intResidents + 1
                dblSum = dblSum + aTPR(1, intCol)
            Next intCol
            
            ' // Set the probabilities of transitioning to an unoccupied state to zero
            For intCol = 1 To intResidents + 1
                aTPR(1, intCol) = 0
            Next intCol
            
            ' // If there is zero probability of transitioning to an occupied state then force
            ' // the transition to the first occupied and active state i.e. '11'
            If (1 - dblSum) <= 0 Then
                aTPR(1, (intResidents + 1) + 2) = 1
            ' // Otherwise proportionally adjust the remaining transition probabilities
            Else
                For intCol = intResidents + 2 To intPossibleStates
                    aTPR(1, intCol) = aTPR(1, intCol) / (1 - dblSum)
                Next intCol
            End If
        End If
        
        ' // Check that this is not a transition probability 'dead-end' i.e. the row is not
        ' // all zeros (occurs on a small number of states, corresponding to the start and
        ' // end times of the time-use survey diaries i.e. 4am)
        dblSum = 0
        For intCol = 1 To intPossibleStates
            dblSum = dblSum + aTPR(1, intCol)
        Next intCol
        
        ' // If the row is all zeros, arbitrarily force the next transition into the lowest
        ' // state
        If dblSum = 0 Then
            aTPR(1, 1) = 1
        End If
        
        ' // Having got the appropriate transition probabilities, determine the state for
        ' // the next time step
        
        ' // Pick a random number
        dblRand = Rnd()
        
        ' // Reset the cumulative probability
        dblCumulativeP = 0
                    
        ' // Cycle through the probabilities for this state
        For intCol = 1 To intPossibleStates
            
            ' // Add the probability
            dblCumulativeP = dblCumulativeP + aTPR(1, intCol)
            
            ' // See if this is a state transition
            If dblRand < dblCumulativeP Then
                ' // This is the combined state for the next time step
                strCombinedState = aTPM(1, intCol + 2)
                
                Exit For
            End If
        Next intCol
        
        ' // Assign combined state to combined state array
        aCombinedState(intTimeStep, 0) = strCombinedState
        aActiveOccupancy(intTimeStep, 0) = WorksheetFunction.Min(CInt(Left(strCombinedState, 1)), CInt(Right(strCombinedState, 1)))
    Next intTimeStep

End Sub




' ===============================================================================================
' ## CalculateThermalGains
'
' ===============================================================================================
Public Sub CalculateThermalGains()
    
    Dim intRow As Integer
    Dim intActiveGains As Integer ' // per active occupant (W)
    Dim intDormantGains As Integer ' per dormant occupant (W)
    Dim intOccupants As Integer
    Dim intActive As Integer
    Dim intActiveOccupants As Integer
    
    intActiveGains = 147
    intDormantGains = 84
    
    For intRow = 0 To 143
        ' // Get number of people asleep in this time step (note: they are not necessarily at home)
        intOccupants = CInt(Left(aCombinedState(intRow, 0), 1))
        
        ' // Get number of active residents (again they are not necessarily at home)
        intActive = CInt(Right(aCombinedState(intRow, 0), 1))
        
        ' // Get number of active occupants (active and at home)
        intActiveOccupants = aActiveOccupancy(intRow, 0)
        
        ' // Calculate thermal gains
        aOccupancyThermalGains(intRow, 0) = intDormantGains * WorksheetFunction.Max(0, intOccupants - intActive) + intActiveGains * intActiveOccupants
    Next intRow
End Sub



' ===============================================================================================
' ## WriteOccupancy
'
' ===============================================================================================
Public Sub WriteOccupancy(dwellingIndexRowOffset As Long)
    Dim intMinute As Integer
    Dim intOccupancy As Integer
    Dim intActivity As Integer
    Dim intOffset As Integer
    
    ' // Get the row offset
    intOffset = 6
    
    ' // Loop through each minute of the day
    For intMinute = 1 To 1440
    
        ' // Get and the occupancy and activity variables for this minute
        intOccupancy = CInt(Left(aCombinedState(((intMinute - 1) \ 10), 0), 1))
        intActivity = CInt(Right(aCombinedState(((intMinute - 1) \ 10), 0), 1))
        
        ' // Write the variables to the results worksheet
        wsResultsDisaggregated.Range("D" + CStr(intMinute + intOffset)).Offset(dwellingIndexRowOffset, 0) = intOccupancy
        wsResultsDisaggregated.Range("E" + CStr(intMinute + intOffset)).Offset(dwellingIndexRowOffset, 0) = intActivity
    Next intMinute
        
End Sub


