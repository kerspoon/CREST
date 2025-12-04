Attribute VB_Name = "clsPortableLCG"
Attribute VB_Base = "0{FCFB3D2A-A0FA-1068-A738-08002B3371B5}"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Attribute VB_TemplateDerived = False
Attribute VB_Customizable = False
'===============================================================================
' clsPortableLCG - Portable Linear Congruential Generator
'===============================================================================
' Cross-platform RNG that produces identical sequences in both VBA and Python.
'
' Uses parameters from Numerical Recipes:
'   - Multiplier (a) = 1664525
'   - Increment (c) = 1013904223
'   - Modulus (m) = 2^32 = 4294967296
'
' Formula: next = (a * current + c) mod m
' Output: next / m to get Double in [0, 1)
'
' IMPORTANT: This must be bit-identical to the Python implementation.
'===============================================================================

Option Explicit

' LCG parameters (Numerical Recipes)
Private Const MULTIPLIER As Long = 1664525
Private Const INCREMENT As Long = 1013904223
' Note: VBA doesn't have unsigned 32-bit, so we use Double for MODULUS
Private Const MODULUS As Double = 4294967296#  ' 2^32

' Current state
Private m_State As Double
Private m_InitialSeed As Double

'===============================================================================
' Initialization
'===============================================================================

Public Sub Initialize(Optional ByVal Seed As Long = 1)
    '
    ' Initialize the LCG with a seed.
    '
    ' Parameters:
    '   Seed - Initial seed value (default: 1)
    '

    ' Convert to unsigned 32-bit by taking modulo
    If Seed < 0 Then
        m_State = CDbl(Seed) + MODULUS
    Else
        m_State = CDbl(Seed)
    End If

    ' Ensure in range [0, MODULUS)
    m_State = ModuloDouble(m_State, MODULUS)
    m_InitialSeed = m_State
    
End Sub

'===============================================================================
' Random Number Generation
'===============================================================================

Public Function Random() As Double
    '
    ' Generate the next random Double in [0.0, 1.0).
    '
    ' Returns: Random number between 0.0 (inclusive) and 1.0 (exclusive)
    '

    ' Update state: next = (a * current + c) mod m
    ' Must handle overflow carefully since VBA doesn't have unsigned types
    m_State = ModuloDouble(MULTIPLIER * m_State + INCREMENT, MODULUS)
    
    DebugLog_Random ("r " & (m_State / MODULUS))

    ' Convert to Double in [0, 1)
    Random = m_State / MODULUS
End Function

Public Function RandomInt(ByVal LowVal As Long, ByVal HighVal As Long) As Long
    '
    ' Generate a random integer in [LowVal, HighVal).
    '
    ' Parameters:
    '   LowVal - Lowest integer (inclusive)
    '   HighVal - Highest integer (exclusive)
    '
    ' Returns: Random integer in [LowVal, HighVal)
    '

    Dim RangeSize As Long

    If LowVal >= HighVal Then
        Err.Raise vbObjectError + 1, "clsPortableLCG", _
                  "LowVal must be less than HighVal"
    End If

    RangeSize = HighVal - LowVal
    RandomInt = LowVal + Int(Random() * RangeSize)
End Function

Public Sub Reset()
    '
    ' Reset the generator to its initial seed.
    '
    m_State = m_InitialSeed
End Sub

'===============================================================================
' Helper Functions
'===============================================================================

Private Function ModuloDouble(ByVal Value As Double, ByVal Divisor As Double) As Double
    '
    ' Compute Value mod Divisor for positive doubles.
    ' VBA's Mod operator only works with integers, so we need this helper.
    '
    ' This is equivalent to: Value - Floor(Value / Divisor) * Divisor
    ' But we use a simpler approach for positive values.
    '

    ModuloDouble = Value - Int(Value / Divisor) * Divisor

    ' Handle floating point errors that might make result negative or >= Divisor
    If ModuloDouble < 0 Then
        ModuloDouble = ModuloDouble + Divisor
    ElseIf ModuloDouble >= Divisor Then
        ModuloDouble = ModuloDouble - Divisor
    End If
End Function

'===============================================================================
' Property Getters
'===============================================================================

Public Property Get State() As Double
    State = m_State
End Property

Public Property Get InitialSeed() As Double
    InitialSeed = m_InitialSeed
End Property
