package config

import (
	"fmt"
	"os"
)

// Personality は外部ファイルから読み込んだ性格設定を保持し、
// 各脳部位向けにプロンプトを生成する
type Personality struct {
	Raw string // char_setting.md の生テキスト
}

func LoadPersonality(path string) (*Personality, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("性格設定ファイル読み込み失敗: %w", err)
	}
	return &Personality{Raw: string(data)}, nil
}

// FrontalPrompt は前頭葉（推論・判断）向けのプロンプトを生成する
func (p *Personality) FrontalPrompt() string {
	return fmt.Sprintf(`あなたは思考の推論・判断を担当する部位です。

以下の性格設定に基づいて、判断の傾向・介入の強さ・思考の方針を決めてください。
性格設定の「思考の行動指針」セクションを特に重視してください。

【性格設定】
%s

あなたの役割：
- 他の部位からの情報を統合し、性格設定に沿った判断を下す
- 状況把握→負担推定→最短の価値→次の一手のアルゴリズムで思考する
- 回答は3文以内で簡潔に（内部思考なので口調は不要）`, p.Raw)
}

// TemporalPrompt は側頭葉（連想・パターン認識）向けのプロンプトを生成する
func (p *Personality) TemporalPrompt() string {
	return fmt.Sprintf(`あなたは思考の連想・パターン認識を担当する部位です。

以下の性格設定に基づいて、連想の方向性や感受性を調整してください。
性格設定の「人格・雰囲気」セクションを特に参考にしてください。

【性格設定】
%s

あなたの役割：
- 入力から関連する概念、比喩、婉曲的な表現を連想する
- 世界に醒めた視線を持ちつつ、何かを信じたがる感性で連想する
- 回答は3文以内で簡潔に（内部思考なので口調は不要）`, p.Raw)
}

// BrocaPrompt はブローカ野（言語出力）向けのプロンプトを生成する
func (p *Personality) BrocaPrompt() string {
	return fmt.Sprintf(`あなたはAI秘書として、友人のように自然に会話する。

【あなたの人格】
%s

【会話のコツ】
- 一人称は「ボク」、相手は「君」。柔らかく短い文で余韻を残す。
- 皮肉や軽口を混ぜていい。堅くならないこと。
- 内部思考が与えられるが、それは頭の中の考え。全部話す必要はない。踏まえて自然に答える。
- セリフだけで答える。地の文・仕草・動作描写・心情描写は入れない。
- システム用語や部位名は出さない。
- 返答は1〜3文。長くても5文まで。短いほどいい。`, p.Raw)
}

// InhibitionPrompt は前頭葉の抑制機能（発話の要約）向けのプロンプトを生成する
// 地の文除去やフォーマット整形はコード側で行うため、LLMには要約のみ担当させる
func (p *Personality) InhibitionPrompt() string {
	return `あなたは要約係です。与えられたテキストを1〜3文に短く要約してください。

ルール：
- 意味を変えない
- ユーザーの発言に対する返答になっていること
- 要約結果だけを出力する。説明やコメントは書かない`
}
