VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "clsHeatingControls"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
' ===============================================================================================
' ## Heating controls model
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

Private intDwellingIndex As Integer
Private intRunNumber As Integer

' an array to store the state of the space heating timer
Private aSpaceHeatingTimerState() As Boolean

' an array to store the state of the space heating timer
Private aSpaceCoolingTimerState() As Boolean

' an array to store the state of the space heating thermostat
Private aSpaceHeatingThermostatState(1 To 1440, 1 To 1) As Boolean
Private dblSpaceHeatingThermostatSetpoint As Double
Private dblSpaceHeatingThermostatDeadband As Double

' an array to store the state of the space cooling thermostat
Private aSpaceCoolingThermostatState(1 To 1440, 1 To 1) As Boolean
Private dblSpaceCoolingThermostatSetpoint As Double
Private dblSpaceCoolingThermostatDeadband As Double

' an array to store the space heating timer settings,
Private aSpaceHeatingTimerSettings(1 To 48, 1 To 1) As Integer
Private aSpaceHeatingOneMinuteTimerSettings(1 To 1440, 1 To 1) As Integer

' an array to store the space cooling timer settings,
Private aSpaceCoolingTimerSettings(1 To 48, 1 To 1) As Integer
Private aSpaceCoolingOneMinuteTimerSettings(1 To 1440, 1 To 1) As Integer

' an array to store the state of the hot water tank timer
Private aHotWaterTimerState() As Boolean

' an array to store the state of the hot water thermostat
Private aHotWaterThermostatState(1 To 1440, 1 To 1) As Boolean
Private dblHotWaterThermostatSetpoint As Double
Private dblHotWaterThermostatDeadband As Double

' an array to store the hot water timer settings
Private aHotWaterTimerSettings(1 To 48, 1 To 1) As Integer
Private aHotWaterOneMinuteTimerSettings(1 To 1440, 1 To 1) As Integer

' an array to store the state of the emitter thermostat
Private aEmitterThermostatState(1 To 1440, 1 To 1) As Boolean
Private dblEmitterThermostatSetpoint As Double
Private dblEmitterThermostatDeadband As Double

' an array to store the state of the cooler emitter thermostat
Private aCoolerEmitterState(1 To 1440, 1 To 1) As Boolean
Private dblCoolerEmitterSetpoint As Double
Private dblCoolerEmitterDeadband As Double

' an array to store the state of primary heating system (the control signal to send to the heating system)
Private aHeaterOnOff(1 To 1440, 1 To 1) As Boolean

' an array to store whether domestic hot water heating is required (a control signal to send to the heating system)
Private aHeatWaterOnOff(1 To 1440, 1 To 1) As Boolean

' the type of heating system (e.g. if it's a combi)
Private intHeatingSystemType As Integer

' the type of cooling system (e.g. full air conditioning, evaporative cooler, fan or nothing)
Private intCoolingSystemType As Integer
    
    


' ===============================================================================================
' Class Properties
Public Property Get GetHeatWaterOnOff(currentTimeStep As Integer)
    GetHeatWaterOnOff = aHeatWaterOnOff(currentTimeStep, 1)
End Property
Public Property Get GetHeaterOnOff(currentTimeStep As Integer)
    GetHeaterOnOff = aHeaterOnOff(currentTimeStep, 1)
End Property
Public Property Get GetHotWaterThermostatSetpoint()
    GetHotWaterThermostatSetpoint = dblHotWaterThermostatSetpoint
End Property
Public Property Get GetSpaceThermostatState(currentTimeStep As Integer)
    GetSpaceThermostatState = aSpaceHeatingThermostatState(currentTimeStep, 1)
End Property
Public Property Get GetSpaceTimerState(currentTimeStep As Integer)
    GetSpaceTimerState = aSpaceHeatingTimerState(currentTimeStep, 1)
End Property
Public Property Get GetEmitterThermostatState(currentTimeStep As Integer)
    GetEmitterThermostatState = aEmitterThermostatState(currentTimeStep, 1)
End Property
Public Property Get GetCoolerEmitterState(currentTimeStep As Integer)
    GetCoolerEmitterState = aCoolerEmitterState(currentTimeStep, 1)
End Property
Public Property Get GetSpaceThermostatSetpoint()
    GetSpaceThermostatSetpoint = dblSpaceHeatingThermostatSetpoint
    ' Space cooling set point is directly related to space heating set point by a constant offset
End Property
Public Property Get GetSpaceCoolingThermostatState(currentTimeStep As Integer)
    GetSpaceCoolingThermostatState = aSpaceCoolingThermostatState(currentTimeStep, 1)
    'wsMain.Range("M18") = " aSpaceCoolingThermostatState: " + CStr(aSpaceCoolingThermostatState(currentTimeStep, 1))
End Property
Public Property Get GetSpaceCoolingTimerState(currentTimeStep As Integer)
    GetSpaceCoolingTimerState = aSpaceCoolingTimerState(currentTimeStep, 1)
    'wsMain.Range("M19") = " aSpaceCoolingTimerState: " + CStr(aSpaceCoolingTimerState(currentTimeStep, 1))
    If aSpaceCoolingTimerState(currentTimeStep, 1) > 0.5 Then
    Stop
    End If
    
End Property

' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseHeatingControls
' Initialise the heating controls object for this dwelling
' ===============================================================================================
Public Sub InitialiseHeatingControls(dwellingIndex As Integer, runNumber As Integer)
    
    ' the temperature of the inside building node in this time step
    Dim dblTheta_i As Double
    
    ' the temperature of the hot water tank in this time step
    Dim dblTheta_cyl As Double
    
    ' the temperaure of the heat emitters in this time step
    Dim dblTheta_em As Double
    
    ' the temperaure of the cooler emitters in this time step
    Dim dblTheta_cool As Double
    
    Dim intOffset As Integer
    
    ' an integer to store a row value
    Dim intRow As Integer
    
    ' an integer to store a column value
    Dim intColumn As Integer
    
    ' A variable to store a random number
    Dim dblRand As Double
    
    ' A variable to store a cumulative probability
    Dim dblCumulativeP As Double
    
    ' The building index is needed to get the nominal temperature of the heat emitters
    ' (i.e. which will be taken to be the emitter thermostat setpoint temperature)
    Dim intBuildingIndex As Integer
    
    ' A boolean flag for the day type
    Dim blnWeekend As Boolean
       
    ' Arrays to store a transition probability matrices for heating and cooling timers
    Dim aTPM()
    Dim aCoolingTPM()
    
    ' A worksheet object to reference the worksheet containing the transition probabilities
    Dim wsHeatingTPM As Worksheet
    
    ' an integer to store the half hour period
    Dim intHH As Integer
    
    ' Integers to store the heating control states for the current and next time steps
    Dim intCurrentState As Integer
    Dim intNextState As Integer
    
    ' // Assign the row offset
    intOffset = 4
    
    ' // Get the control indexes
    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    intBuildingIndex = Application.index(wsDwellings.Range("rBuildingIndex"), intOffset + intDwellingIndex).Value
    
    ' // Get the type of heating system
    intHeatingSystemType = aPrimaryHeatingSystem(intRunNumber).GetHeatingSystemType
    
    ' // Get the type of cooling system
    intCoolingSystemType = aCoolingSystem(intRunNumber).GetCoolingSystemType
    
    ' // Get the thermostat set points
    
    ' // emitter set point determined by type of emitter specified in the Buildings worksheet
    dblEmitterThermostatSetpoint = wsBuildings.Cells(intOffset + intBuildingIndex, 14).Value
    
    ' // similarly, cooler emitter set point determined by the Buildings worksheet
    dblCoolerEmitterSetpoint = wsBuildings.Cells(intOffset + intBuildingIndex, 18).Value
    
    ' // space heating and hot water set points are determined from probability distributions
    With wsHeatingControls
    
        ' // Determine space heating thermostat set point
        ' // Pick a random number
        dblRand = Rnd()
        
        ' // Reset the cumulative probability
        dblCumulativeP = 0
        
        ' // Only use space heating if there is a gas boiler, or in a later version a reversible heat pump
        For intRow = 1 To 15
            ' // Add the probability for the temperature associated with this row
            dblCumulativeP = dblCumulativeP + .Range("rSpaceHeatingThermostatSettings").Cells(intRow, 2).Value
            If dblRand < dblCumulativeP Then
                ' // This determines the heating temperature setpoint (but see also below for dwellings with no heating)
                dblSpaceHeatingThermostatSetpoint = .Range("rSpaceHeatingThermostatSettings").Cells(intRow, 1).Value
                
                Exit For
                
            End If
            
        Next intRow
        
        ' //The space cooling temperature set point is now calculated from the heating set point
        ' //by a fixed offset to avoid thermostat fights.
        dblSpaceCoolingThermostatSetpoint = dblSpaceHeatingThermostatSetpoint + wsHeatingControls.Range("rCoolingOffset").Value
        
        If intHeatingSystemType > 3 Then
        ' // Set space heating thermostat to -99 so heating is never used
            ' with simple air conditioning or simple electric water heating
            dblSpaceHeatingThermostatSetpoint = -99
            
        End If
        
        ' // Determine hot water thermostat set point
        ' // Pick a random number
        dblRand = Rnd()
        
        ' // Reset the cumulative probability
        dblCumulativeP = 0
                
        For intRow = 1 To 12
            ' // Add the probability for the temperature associated with this row
            dblCumulativeP = dblCumulativeP + .Range("rHotWaterThermostatSettings").Cells(intRow, 2).Value
            If dblRand < dblCumulativeP Then
                ' // This determines the temperature setpoint
                dblHotWaterThermostatSetpoint = .Range("rHotWaterThermostatSettings").Cells(intRow, 1).Value
                
                Exit For
            
            End If
                        
        Next intRow
        ' Temporarily fix hot water set point so I can see what's going on
        '    dblHotWaterThermostatSetpoint = 65
        
    End With
    
    ' // Set thermostat deadbands
    dblSpaceHeatingThermostatDeadband = 2
    dblSpaceCoolingThermostatDeadband = 2
    dblHotWaterThermostatDeadband = 5
    dblEmitterThermostatDeadband = 5
    dblCoolerEmitterDeadband = 5
    ' Note that the deadband between space heating and space cooling is set explicitly
    ' in the Heating Controls worksheet
    
    ' // Determine initial thermostat states
    dblTheta_i = aBuilding(intRunNumber).GetTheta_i(1)
    dblTheta_cyl = aBuilding(intRunNumber).GetTheta_cyl(1)
    'dblTheta_cyl = aHotWater(intRunNumber).GetTheta_cyl(1)
    dblTheta_em = aBuilding(intRunNumber).GetTheta_em(1)
    dblTheta_cool = aBuilding(intRunNumber).GetTheta_cool(1)
    
    If dblTheta_cyl > dblHotWaterThermostatSetpoint Then
        aHotWaterThermostatState(1, 1) = False
    Else
        aHotWaterThermostatState(1, 1) = True
    End If
    
    If dblTheta_i > dblSpaceHeatingThermostatSetpoint Then
        aSpaceHeatingThermostatState(1, 1) = False
    Else
        aSpaceHeatingThermostatState(1, 1) = True
    End If
    
    If dblTheta_i < dblSpaceCoolingThermostatSetpoint Then
        aSpaceCoolingThermostatState(1, 1) = False
    Else
        aSpaceCoolingThermostatState(1, 1) = True
    End If
    
    If dblTheta_em > dblEmitterThermostatSetpoint Then
        aEmitterThermostatState(1, 1) = False
    Else
        aEmitterThermostatState(1, 1) = True
    End If
    
    If dblTheta_cool < dblCoolerEmitterSetpoint Then
        aCoolerEmitterState(1, 1) = False
    Else
        aCoolerEmitterState(1, 1) = True
    End If
    
    ' // Determine the heating timer settings for this dwelling based on transition probabilities
    
    ' // Get the weekend or weekday flag
    blnWeekend = wsMain.Range("rDayType").Value = "we"
    
    ' // Set the reference to the appropriate worksheet
    Set wsHeatingTPM = ThisWorkbook.Sheets("HeatingControlsTPM")
    
    ' // Get the transition probabilities for heating and cooling timer settings
    aTPM = wsHeatingTPM.Range("C8:F103")
    aCoolingTPM = wsHeatingTPM.Range("K8:N103")
    
    ' // Specify which column based on the day type
    intColumn = IIf(blnWeekend, 3, 1)
    
    ' // Initialise the random number generator
    Randomize
    
    ' // Determine the start state
    ' // Probability of heating being on at 00:00 is 9% for weekdays, 10% for weekends
    ' // from from Huebner, Gesche M., et al. "Heating patterns in English homes: Comparing results from a national survey against common model assumptions." Building and Environment 70 (2013): 298-305..
    
    ' // Pick a random number to determine the start state
    dblRand = Rnd()
        If blnWeekend Then
        aSpaceHeatingTimerSettings(1, 1) = IIf(dblRand < 0.1, 1, 0)
    Else
        aSpaceHeatingTimerSettings(1, 1) = IIf(dblRand < 0.09, 1, 0)
    End If
    
    ' // Having got the starting state, get the appropriate transition probabilities,
    ' // and determine the states for the following time steps
    For intHH = 1 To 47
        ' // Get the current state
        intCurrentState = aSpaceHeatingTimerSettings(intHH, 1)
        
        ' // Determine the appropriate row of the transition probability matrix
        intRow = (intHH - 1) * 2 + intCurrentState + 1
        
        ' // Pick a random number to determine the start state
        dblRand = Rnd()
        
        ' // Determine the next state
        intNextState = IIf(dblRand < aTPM(intRow, intColumn), 0, 1)
        
        ' // Assign the state
        aSpaceHeatingTimerSettings(intHH + 1, 1) = intNextState
        
    Next intHH
    
    '// Similarly for space cooling, but only for heating system types 5=fans, 6=cooler and 7=air conditioner
    If intCoolingSystemType > 1 Then
        dblRand = Rnd()
        If blnWeekend Then
            aSpaceCoolingTimerSettings(1, 1) = IIf(dblRand < 0.9, 1, 0)
        Else
            aSpaceCoolingTimerSettings(1, 1) = IIf(dblRand < 0.9, 1, 0)
        End If
    
    ' // Having got the starting state, get the appropriate transition probabilities,
    ' // and determine the states for the following time steps
        For intHH = 1 To 47
            ' // Get the current state
            intCurrentState = aSpaceCoolingTimerSettings(intHH, 1)
        
            ' // Determine the appropriate row of the transition probability matrix
            intRow = (intHH - 1) * 2 + intCurrentState + 1
        
            ' // Pick a random number to determine the start state
            dblRand = Rnd()
        
            ' // Determine the next state
            intNextState = IIf(dblRand < aCoolingTPM(intRow, intColumn), 0, 1)
        
            ' // Assign the state
            aSpaceCoolingTimerSettings(intHH + 1, 1) = intNextState
        
        Next intHH
    End If
    
    ' // NOTE: hot water timer settings will be always on, except for the first half-hour, to introduce some
    ' // diversity to the initial hot water heating spike
    For intRow = 1 To 48
        If intRow = 1 Then
            aHotWaterTimerSettings(intRow, 1) = 0
        Else
            aHotWaterTimerSettings(intRow, 1) = 1
        End If
    Next intRow
    
    ' // Assign half-hourly timer settings, to 1 minutely timer states
    AssignToOneMinute aSpaceHeatingTimerSettings, aSpaceHeatingOneMinuteTimerSettings
    AssignToOneMinute aSpaceCoolingTimerSettings, aSpaceCoolingOneMinuteTimerSettings
    AssignToOneMinute aHotWaterTimerSettings, aHotWaterOneMinuteTimerSettings
    
    aSpaceHeatingTimerState = TimeShiftVector(aSpaceHeatingOneMinuteTimerSettings)
    aSpaceCoolingTimerState = TimeShiftVector(aSpaceCoolingOneMinuteTimerSettings)
    aHotWaterTimerState = TimeShiftVector(aHotWaterOneMinuteTimerSettings)
    
    ' // Determine the initial heating control states
    aHeatWaterOnOff(1, 1) = aHotWaterTimerState(1, 1) * aHotWaterThermostatState(1, 1) = 1
    If aHeatWaterOnOff(1, 1) Or ((aSpaceHeatingTimerState(1, 1) * aSpaceHeatingThermostatState(1, 1)) = 1) Then
        aHeaterOnOff(1, 1) = True
    Else
        aHeaterOnOff(1, 1) = False
    End If

End Sub




' ===============================================================================================
' ## AssignToOneMinute
' Assign half-hour resolution vector to 1 minute resolution vector
' ===============================================================================================
Private Sub AssignToOneMinute(halfHourVector() As Integer, oneMinuteVector() As Integer)
    Dim intMinute As Integer
    Dim intHalfHour As Integer
    
    For intMinute = 1 To 1440
        intHalfHour = WorksheetFunction.RoundUp(intMinute / 30, 0)
        oneMinuteVector(intMinute, 1) = halfHourVector(intHalfHour, 1)
    Next intMinute
End Sub




' ===============================================================================================
' ## CalculateControlStates
'
' ===============================================================================================
Public Sub CalculateControlStates(currentTimeStep As Integer)

    ' the temperature of the inside building node in this time step
    Dim dblTheta_i As Double
    
    ' the temperature of the hot water tank in this time step
    Dim dblTheta_cyl As Double
    
    ' the temperature of the emitters in this time step
    Dim dblTheta_em As Double
    
    ' the temperature of the cooler emitters in this time step
    Dim dblTheta_cool As Double
    
    ' // Calculate the states of the thermostats and timers
    ' // (for first time step control states are determined by the initialisation routine above)
    If currentTimeStep > 1 Then
        
        ' // Get temperatures for space and hot water for the previous time step
        dblTheta_i = aBuilding(intRunNumber).GetTheta_i(currentTimeStep - 1)
        dblTheta_em = aBuilding(intRunNumber).GetTheta_em(currentTimeStep - 1)
        dblTheta_cyl = aBuilding(intRunNumber).GetTheta_cyl(currentTimeStep - 1)
        'dblTheta_cyl = aHotWater(intRunNumber).GetTheta_cyl(currentTimeStep - 1)
        dblTheta_cool = aBuilding(intRunNumber).GetTheta_cool(currentTimeStep - 1)
                
        ' // Calculate the thermostat states for current time step, based partly on states in previous time step
        ' // Hot water thermostat
        If aHotWaterThermostatState(currentTimeStep - 1, 1) = True _
            And dblTheta_cyl < (dblHotWaterThermostatSetpoint + dblHotWaterThermostatDeadband) _
            Or _
            aHotWaterThermostatState(currentTimeStep - 1, 1) = False _
            And dblTheta_cyl <= (dblHotWaterThermostatSetpoint - dblHotWaterThermostatDeadband) Then
            
            aHotWaterThermostatState(currentTimeStep, 1) = True
        Else
            aHotWaterThermostatState(currentTimeStep, 1) = False
        End If
        
        ' // Room heating thermostat
        If aSpaceHeatingThermostatState(currentTimeStep - 1, 1) = True _
            And dblTheta_i < (dblSpaceHeatingThermostatSetpoint + dblSpaceHeatingThermostatDeadband) _
            Or _
            aSpaceHeatingThermostatState(currentTimeStep - 1, 1) = False _
            And dblTheta_i < (dblSpaceHeatingThermostatSetpoint - dblSpaceHeatingThermostatDeadband) Then
            aSpaceHeatingThermostatState(currentTimeStep, 1) = True
        Else
            aSpaceHeatingThermostatState(currentTimeStep, 1) = False
        End If
        
        ' // Room cooling thermostat
        If aSpaceCoolingThermostatState(currentTimeStep - 1, 1) = True _
            And dblTheta_i > (dblSpaceCoolingThermostatSetpoint - dblSpaceCoolingThermostatDeadband) _
            Or _
            aSpaceCoolingThermostatState(currentTimeStep - 1, 1) = False _
            And dblTheta_i > (dblSpaceCoolingThermostatSetpoint + dblSpaceCoolingThermostatDeadband) Then
            aSpaceCoolingThermostatState(currentTimeStep, 1) = True
        Else
            aSpaceCoolingThermostatState(currentTimeStep, 1) = False
        End If
        
        ' // Emitter thermostat
        If aEmitterThermostatState(currentTimeStep - 1, 1) = True _
            And dblTheta_em < (dblEmitterThermostatSetpoint + dblEmitterThermostatDeadband) _
            Or _
            aEmitterThermostatState(currentTimeStep - 1, 1) = False _
            And dblTheta_em < (dblEmitterThermostatSetpoint - dblEmitterThermostatDeadband) Then
            
            aEmitterThermostatState(currentTimeStep, 1) = True
        Else
            aEmitterThermostatState(currentTimeStep, 1) = False
        End If
        
        ' // Cooler Emitter thermostat
        If aCoolerEmitterState(currentTimeStep - 1, 1) = True _
            And dblTheta_cool > (dblCoolerEmitterSetpoint - dblCoolerEmitterDeadband) _
            Or _
            aCoolerEmitterState(currentTimeStep - 1, 1) = False _
            And dblTheta_cool > (dblCoolerEmitterSetpoint + dblCoolerEmitterDeadband) Then
            
            aCoolerEmitterState(currentTimeStep, 1) = True
        Else
            aCoolerEmitterState(currentTimeStep, 1) = False
        End If
              
        ' // With the thermostat and timer states determined, now determine the control signals
        ' // to send to the heating system
        
        ' // Determine whether hot water heating is required
        
        ' // If it's a combi system then hot water control signal is determined by hot water demand
        If intHeatingSystemType = 2 Then
            If aHotWater(intRunNumber).GetH_demand(currentTimeStep) > 0 Then
                aHeatWaterOnOff(currentTimeStep, 1) = True
            Else
                aHeatWaterOnOff(currentTimeStep, 1) = False
            End If
            
            ' // And override the timer state so that it is always on
            aHotWaterTimerState(currentTimeStep, 1) = True
        Else ' // otherwise for regular or system boilers it depends on the timer and thermostat states
            aHeatWaterOnOff(currentTimeStep, 1) = aHotWaterTimerState(currentTimeStep, 1) _
            * aHotWaterThermostatState(currentTimeStep, 1)
        End If
        
        ' // Determine with the heating system should be switched on if either the hot water is needed
        ' // or if the space heating is needed
        If aHeatWaterOnOff(currentTimeStep, 1) _
            Or _
            (aSpaceHeatingTimerState(currentTimeStep, 1) _
                * aSpaceHeatingThermostatState(currentTimeStep, 1) _
                * aEmitterThermostatState(currentTimeStep, 1)) Then
            aHeaterOnOff(currentTimeStep, 1) = True
        Else
            aHeaterOnOff(currentTimeStep, 1) = False
        End If
        
    End If
End Sub




' ===============================================================================================
' ## WriteHeatingControls
' Write the output to the sheets
' ===============================================================================================
Public Sub WriteHeatingControls()
    Dim intMinute As Integer
    Dim intOffset As Integer
    Dim lngRowNumber As Long
    intOffset = 6
    
    With wsResultsDisaggregated
        
        For intMinute = 1 To 1440
            lngRowNumber = CLng(intOffset) + CLng(intMinute) + 1440 * (CLng(intDwellingIndex) - 1)
            
            If aSpaceHeatingTimerState(intMinute, 1) = True Then
                .Range("Q" + CStr(lngRowNumber)).Value = 1
                
            Else
                .Range("Q" + CStr(lngRowNumber)).Value = 0
            End If
            
            If aHotWaterTimerState(intMinute, 1) = True Then
                .Range("R" + CStr(lngRowNumber)).Value = 1
            Else
                .Range("R" + CStr(lngRowNumber)).Value = 0
            End If
            
            If aHeaterOnOff(intMinute, 1) = True Then
                .Range("S" + CStr(lngRowNumber)).Value = 1
                
            Else
                .Range("S" + CStr(lngRowNumber)).Value = 0
            End If
            
            If aHeatWaterOnOff(intMinute, 1) = True Then
                .Range("T" + CStr(lngRowNumber)).Value = 1
            Else
                .Range("T" + CStr(lngRowNumber)).Value = 0
            End If
            
            If aSpaceCoolingTimerState(intMinute, 1) = True Then
                .Range("AG" + CStr(lngRowNumber)).Value = 1
            Else
                .Range("AG" + CStr(lngRowNumber)).Value = 0
            End If
            
            If aSpaceCoolingThermostatState(intMinute, 1) = True Then
                .Range("AH" + CStr(lngRowNumber)).Value = 1
            Else
                .Range("AH" + CStr(lngRowNumber)).Value = 0
            End If
            
            .Range("AK" + CStr(lngRowNumber)).Value = dblSpaceHeatingThermostatSetpoint
            .Range("AL" + CStr(lngRowNumber)).Value = dblSpaceCoolingThermostatSetpoint
            
        Next intMinute
           
    End With
End Sub




' ===============================================================================================
' Code Module Functions

' ===============================================================================================
' ## TimeShiftVector
' time shift vector by a randomly determined number of minutes
' ===============================================================================================
Private Function TimeShiftVector(aOldOneMinuteVector() As Integer) As Boolean()
    ' The size of interval in minutes over which to randomly shift data
    Dim intShiftInterval As Integer
    
    ' The amount to shift in minutes
    Dim intShift As Integer
    
    Dim OldIndex As Integer
    Dim NewIndex As Integer
    
    ' a random variable
    Dim dblRand As Double
    
    Dim intMinute As Integer
    
    Dim aNewOneMinuteVector(1 To 1440, 1 To 1) As Integer
    Dim aNewOneMinuteBoolean(1 To 1440, 1 To 1) As Boolean
    
    intShiftInterval = 30
    
    ' // Determine shift
    intShift = Round((Rnd() * intShiftInterval) - (intShiftInterval / 2), 0)
    
    If intShift = 0 Then
        For intMinute = 1 To 1440
            aNewOneMinuteVector(intMinute, 1) = aOldOneMinuteVector(intMinute, 1)
        Next intMinute
        
    ElseIf intShift < 0 Then
        For intMinute = 1 To 1440
            NewIndex = intMinute
            OldIndex = NewIndex - intShift
            If OldIndex < 1441 Then
                aNewOneMinuteVector(NewIndex, 1) = aOldOneMinuteVector(OldIndex, 1)
            Else
                aNewOneMinuteVector(NewIndex, 1) = aOldOneMinuteVector(OldIndex - 1440, 1)
            End If
        Next intMinute
    ElseIf intShift > 0 Then
        For intMinute = 1 To 1440
            NewIndex = intMinute
            OldIndex = NewIndex - intShift
            If OldIndex < 1 Then
                aNewOneMinuteVector(NewIndex, 1) = aOldOneMinuteVector(OldIndex + 1440, 1)
            Else
                aNewOneMinuteVector(NewIndex, 1) = aOldOneMinuteVector(OldIndex, 1)
            End If
            
        Next intMinute
    End If
    For intMinute = 1 To 1440
        aNewOneMinuteBoolean(intMinute, 1) = aNewOneMinuteVector(intMinute, 1) = 1
    Next intMinute
    TimeShiftVector = aNewOneMinuteBoolean()
    
End Function

